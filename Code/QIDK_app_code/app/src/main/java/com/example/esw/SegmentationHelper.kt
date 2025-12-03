package com.example.esw

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import java.util.TreeMap

class SegmentationHelper(private val context: Context) {

    companion object {
        private const val TAG = "SegmentationHelper"
        private const val INPUT_SIZE = 640
        private const val PIXEL_SIZE = 3
        private const val FLOAT_SIZE = 4
    }

    // ===== INTERPRETERS: TFLite GPU, NNAPI, and CPU =====
    private var interpreterGPU: Interpreter? = null
    private var interpreterNNAPI: Interpreter? = null
    private var interpreterCPU: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    // Unavailable reasons for user-visible diagnostics
    private var gpuUnavailableReason: String? = null

    // Current active delegate
    enum class DelegateType { GPU, NNAPI, CPU }
    private var currentDelegate = DelegateType.CPU

    // Delegate availability status
    private var gpuAvailable = false
    private var nnapiAvailable = false
    private var cpuAvailable = false

    private val OUTPUT_0_DIMS = intArrayOf(1, 39, 8400)
    private val OUTPUT_1_DIMS = intArrayOf(1, 160, 160, 32)

    val classLabels = arrayOf("Animal", "Waste", "Person")
    val classColors = mapOf(
        0 to Color.rgb(255, 0, 0),
        1 to Color.rgb(0, 255, 0),
        2 to Color.rgb(0, 0, 255)
    )

    private var scaleX = 1.0f
    private var scaleY = 1.0f
    private var padLeft = 0f
    private var padTop = 0f

    data class SegmentationStats(
        val classStats: Map<String, ClassStats>,
        val totalSegmentedPixels: Int,
        val totalImagePixels: Int,
        val imageCoverage: Float,
        val processingTimeMs: Long,
        val preprocessTimeMs: Long,
        val inferenceTimeMs: Long,
        val postprocessTimeMs: Long,
        val delegateUsed: String
    )

    data class ClassStats(
        val className: String,
        val pixels: Int,
        val coverage: Float,
        val color: Int,
        val objectCount: Int = 0  // Number of individual objects detected
    )

    private var lastStats: SegmentationStats? = null
    private var lastMaskOnlyBitmap: Bitmap? = null
    private var lastSegmentedBitmap: Bitmap? = null

    init {
        try {
            logDeviceInfo()
            val modelFile = loadModelFile(context)

            // Setup TFLite GPU delegate (OpenGL/Vulkan)
            setupTFLiteGpuInterpreter(modelFile)

            // Setup NNAPI (may route to vendor accelerator if available)
            setupNNAPIInterpreter(modelFile)

            // Always setup CPU as fallback
            setupCPUInterpreter(modelFile)

            // Set initial delegate based on availability (prefer TFLite GPU > NNAPI > CPU)
            currentDelegate = when {
                gpuAvailable -> DelegateType.GPU
                nnapiAvailable -> DelegateType.NNAPI
                else -> DelegateType.CPU
            }

            Log.d(TAG, "Initialization complete - GPU: $gpuAvailable, NNAPI: $nnapiAvailable, CPU: $cpuAvailable, Current: $currentDelegate")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize SegmentationHelper", e)
            e.printStackTrace()
        }
    }

    private fun logDeviceInfo() {
        Log.d(TAG, "=== Device Information ===")
        Log.d(TAG, "Manufacturer: ${android.os.Build.MANUFACTURER}")
        Log.d(TAG, "Model: ${android.os.Build.MODEL}")
        Log.d(TAG, "Android Version: ${android.os.Build.VERSION.RELEASE}")
        Log.d(TAG, "SDK: ${android.os.Build.VERSION.SDK_INT}")
        Log.d(TAG, "Hardware: ${android.os.Build.HARDWARE}")
        Log.d(TAG, "Supported ABIs: ${android.os.Build.SUPPORTED_ABIS.joinToString()}")
        Log.d(TAG, "Native library dir: ${context.applicationInfo.nativeLibraryDir}")

        // Check for TensorFlow Lite libraries
        try {
            val libDir = java.io.File(context.applicationInfo.nativeLibraryDir)
            if (libDir.exists()) {
                val tfliteLibs = libDir.list()?.filter {
                    it.contains("tensorflowlite", ignoreCase = true) ||
                    it.contains("tensorflow-lite", ignoreCase = true)
                }
                Log.d(TAG, "TFLite libraries found in ${libDir.absolutePath}: ${tfliteLibs?.joinToString() ?: "NONE"}")
                Log.d(TAG, "Expect to see libtensorflowlite_jni.so for CPU and libtensorflowlite_gpu_jni.so for GPU")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to list libraries", e)
        }
        Log.d(TAG, "========================")
    }


    private fun setupTFLiteGpuInterpreter(modelFile: ByteBuffer) {
        try {
            Log.d(TAG, "Setting up TFLite GPU delegate...")
            
            val compat = CompatibilityList()
            if (!compat.isDelegateSupportedOnThisDevice) {
                gpuAvailable = false
                gpuUnavailableReason = "TFLite GPU not supported on this device"
                Log.w(TAG, "✗ TFLite GPU not supported on this device")
                return
            }

            // Create GPU delegate with default options for maximum compatibility
            gpuDelegate = GpuDelegate()
            
            val interpreterOptions = Interpreter.Options().apply {
                addDelegate(gpuDelegate!!)
                setNumThreads(4)
            }
            
            modelFile.rewind()
            interpreterGPU = Interpreter(modelFile, interpreterOptions)
            gpuAvailable = true
            
            Log.d(TAG, "✓ TFLite GPU Interpreter initialized successfully")
            
            // Warmup to stabilize performance
            try { 
                warmupInterpreter("TFLite GPU", interpreterGPU) 
            } catch (e: Exception) {
                Log.w(TAG, "Warmup on TFLite GPU failed: ${e.message}")
            }
            
        } catch (e: Exception) {
            gpuAvailable = false
            gpuUnavailableReason = "TFLite GPU init failed: ${e.message ?: "unknown"}"
            Log.e(TAG, "✗ Failed o initialize TFLite GPU: ${e.message}")
            Log.w(TAG, "TFLite GPU delegate not working, will fall back to NNAPI or CPU")
            try { gpuDelegate?.close() } catch (_: Exception) {}
            gpuDelegate = null
        }
    }

    private fun setupNNAPIInterpreter(modelFile: ByteBuffer) {
        try {
            Log.d(TAG, "Setting up NNAPI delegate interpreter...")

            // Prefer explicit NNAPI delegate for better control
            nnapiDelegate = try { NnApiDelegate() } catch (e: Throwable) {
                Log.w(TAG, "NNAPI delegate not available: ${e.message}")
                null
            }
            if (nnapiDelegate == null) {
                nnapiAvailable = false
                return
            }

            val options = Interpreter.Options().apply {
                addDelegate(nnapiDelegate!!)
                setNumThreads(4)
            }

            modelFile.rewind()
            interpreterNNAPI = Interpreter(modelFile, options)
            nnapiAvailable = true

            Log.d(TAG, "✓ NNAPI Interpreter initialized successfully")

            // Warmup to stabilize timing
            try { warmupInterpreter("NNAPI", interpreterNNAPI) } catch (e: Exception) {
                Log.w(TAG, "Warmup on NNAPI failed: ${e.message}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "✗ Failed to initialize NNAPI interpreter: ${e.message}")
            nnapiAvailable = false
            try { nnapiDelegate?.close() } catch (_: Exception) {}
            nnapiDelegate = null
        }
    }

    private fun setupCPUInterpreter(modelFile: ByteBuffer) {
        try {
            Log.d(TAG, "Setting up CPU interpreter...")

            val interpreterOptions = Interpreter.Options()
            interpreterOptions.setNumThreads(4)
            interpreterOptions.setUseNNAPI(false) // Disable NNAPI to ensure pure CPU

            modelFile.rewind()
            interpreterCPU = Interpreter(modelFile, interpreterOptions)
            cpuAvailable = true

            Log.d(TAG, "✓ CPU Interpreter initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "✗ Failed to initialize CPU interpreter", e)
            e.printStackTrace()
            cpuAvailable = false
        }
    }


    fun switchDelegate() {
        val oldDelegate = currentDelegate

        currentDelegate = when (currentDelegate) {
            DelegateType.GPU -> when {
                nnapiAvailable -> DelegateType.NNAPI
                cpuAvailable -> DelegateType.CPU
                else -> DelegateType.GPU
            }
            DelegateType.NNAPI -> when {
                cpuAvailable -> DelegateType.CPU
                gpuAvailable -> DelegateType.GPU
                else -> DelegateType.NNAPI
            }
            DelegateType.CPU -> when {
                gpuAvailable -> DelegateType.GPU
                nnapiAvailable -> DelegateType.NNAPI
                else -> DelegateType.CPU
            }
        }

        Log.d(TAG, "Switched delegate from $oldDelegate to $currentDelegate")
    }

    fun forceDelegate(type: DelegateType): Boolean {
        return when (type) {
            DelegateType.GPU -> if (gpuAvailable) { currentDelegate = type; true } else false
            DelegateType.NNAPI -> if (nnapiAvailable) { currentDelegate = type; true } else false
            DelegateType.CPU -> if (cpuAvailable) { currentDelegate = type; true } else false
        }
    }

    fun getCurrentDelegate(): String {
        return when (currentDelegate) {
            DelegateType.GPU -> if (gpuAvailable) "GPU (TFLite)" else "GPU (Not Available)"
            DelegateType.NNAPI -> if (nnapiAvailable) "NPU" else "NPU (Not Available)"
            DelegateType.CPU -> "CPU"
        }
    }

    fun isGPUAvailable(): Boolean = gpuAvailable
    fun isNNAPIAvailable(): Boolean = nnapiAvailable
    fun isCPUAvailable(): Boolean = cpuAvailable

    fun getGpuUnavailableReason(): String? = if (!gpuAvailable) gpuUnavailableReason else null

    private fun loadModelFile(context: Context): ByteBuffer {
        val ASSET_FILE = "yolo11_seg.tflite"
        try {
            val fd = context.assets.openFd(ASSET_FILE)
            FileInputStream(fd.fileDescriptor).use { input ->
                val fileChannel = input.channel
                val buffer = fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                )

                Log.d(TAG, "Model loaded: $ASSET_FILE (${buffer.capacity()} bytes)")
                return buffer
            }
        } catch (e: IOException) {
            throw RuntimeException("Failed to load model '$ASSET_FILE'", e)
        }
    }

    fun runSegmentation(bitmapInput: Bitmap): Bitmap {
        val startTotal = System.currentTimeMillis()
        val bitmap = ensureSoftwareArgb(bitmapInput)

        val startPre = System.currentTimeMillis()
        val inputBuffer = preprocessImage(bitmap)
        val preprocessMs = System.currentTimeMillis() - startPre

        val interpreter = when (currentDelegate) {
            DelegateType.GPU -> if (gpuAvailable) interpreterGPU else null
            DelegateType.NNAPI -> if (nnapiAvailable) interpreterNNAPI else null
            DelegateType.CPU -> interpreterCPU
        }

        if (interpreter == null) {
            Log.e(TAG, "Selected interpreter ($currentDelegate) not available")

            // Fallback to CPU
            if (cpuAvailable && currentDelegate != DelegateType.CPU) {
                Log.w(TAG, "Falling back to CPU")
                currentDelegate = DelegateType.CPU
                return runSegmentation(bitmapInput)
            }

            Log.e(TAG, "No interpreter available!")
            return bitmap
        }

        val outputMap = TreeMap<Int, Any>()

        val detectionsOutput = Array(OUTPUT_0_DIMS[0]) {
            Array(OUTPUT_0_DIMS[1]) { FloatArray(OUTPUT_0_DIMS[2]) }
        }
        val masksOutput = Array(OUTPUT_1_DIMS[0]) {
            Array(OUTPUT_1_DIMS[1]) {
                Array(OUTPUT_1_DIMS[2]) { FloatArray(OUTPUT_1_DIMS[3]) }
            }
        }
        outputMap[0] = detectionsOutput
        outputMap[1] = masksOutput

        val startInf = System.currentTimeMillis()
        try {
            Log.d(TAG, "Running inference on $currentDelegate...")
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed on $currentDelegate", e)
            e.printStackTrace()
            return bitmap
        }
        val inferenceMs = System.currentTimeMillis() - startInf

        val startPost = System.currentTimeMillis()
        val resultBitmap = try {
            postProcessOutputs(bitmap, detectionsOutput[0], masksOutput[0])
        } catch (e: Exception) {
            Log.e(TAG, "Post-processing failed", e)
            e.printStackTrace()
            bitmap
        }
        val postprocessMs = System.currentTimeMillis() - startPost

        val totalMs = System.currentTimeMillis() - startTotal
        val delegateName = getCurrentDelegate()

        lastStats = lastStats?.copy(
            processingTimeMs = totalMs,
            preprocessTimeMs = preprocessMs,
            inferenceTimeMs = inferenceMs,
            postprocessTimeMs = postprocessMs,
            delegateUsed = delegateName
        ) ?: SegmentationStats(
            classStats = emptyMap(),
            totalSegmentedPixels = 0,
            totalImagePixels = bitmap.width * bitmap.height,
            imageCoverage = 0f,
            processingTimeMs = totalMs,
            preprocessTimeMs = preprocessMs,
            inferenceTimeMs = inferenceMs,
            postprocessTimeMs = postprocessMs,
            delegateUsed = delegateName
        )

        Log.d(TAG, "[$delegateName] total=${totalMs}ms (pre=${preprocessMs}ms, inf=${inferenceMs}ms, post=${postprocessMs}ms)")
        return resultBitmap
    }

    fun getLastStats(): SegmentationStats? = lastStats
    fun getLastMaskOnlyBitmap(): Bitmap? = lastMaskOnlyBitmap
    fun getLastSegmentedBitmap(): Bitmap? = lastSegmentedBitmap

    private fun ensureSoftwareArgb(b: Bitmap): Bitmap {
        return try {
            if (b.config == Bitmap.Config.HARDWARE || b.config != Bitmap.Config.ARGB_8888) {
                b.copy(Bitmap.Config.ARGB_8888, true)
            } else b
        } catch (e: Exception) {
            Log.e(TAG, "ensureSoftwareArgb failed", e)
            val out = Bitmap.createBitmap(b.width, b.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(out)
            canvas.drawBitmap(b, 0f, 0f, null)
            out
        }
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE * FLOAT_SIZE)
        byteBuffer.order(ByteOrder.nativeOrder())

        val letterboxBitmap = letterboxResize(bitmap, INPUT_SIZE, INPUT_SIZE)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        letterboxBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val floatBuffer = byteBuffer.asFloatBuffer()

        for (pixelValue in intValues) {
            floatBuffer.put(((pixelValue shr 16 and 0xFF) / 255.0f))
            floatBuffer.put(((pixelValue shr 8 and 0xFF) / 255.0f))
            floatBuffer.put(((pixelValue and 0xFF) / 255.0f))
        }

        try { letterboxBitmap.recycle() } catch (_: Exception) {}
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun letterboxResize(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val src = ensureSoftwareArgb(source)
        val scale = min(targetWidth.toFloat() / src.width, targetHeight.toFloat() / src.height)
        val newWidth = (src.width * scale).toInt()
        val newHeight = (src.height * scale).toInt()
        val scaledBitmap = Bitmap.createScaledBitmap(src, newWidth, newHeight, true)

        val paddedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        padLeft = ((targetWidth - newWidth) / 2).toFloat()
        padTop = ((targetHeight - newHeight) / 2).toFloat()
        scaleX = scale
        scaleY = scale

        canvas.drawBitmap(scaledBitmap, padLeft, padTop, null)

        try { if (scaledBitmap !== src) scaledBitmap.recycle() } catch (_: Exception) {}
        return paddedBitmap
    }

    private fun postProcessOutputs(
        originalBitmap: Bitmap,
        detections: Array<FloatArray>,
        masks: Array<Array<FloatArray>>
    ): Bitmap {
        val detectedObjects = mutableListOf<Detection>()
        val confidenceThreshold = 0.05f
        val iouThreshold = 0.45f
        val numClasses = classLabels.size
        val numCoefficients = 32

        for (i in 0 until 8400) {
            val x = detections[0][i] * INPUT_SIZE
            val y = detections[1][i] * INPUT_SIZE
            val w = detections[2][i] * INPUT_SIZE
            val h = detections[3][i] * INPUT_SIZE

            var maxScore = -1.0f
            var classId = -1
            for (j in 0 until numClasses) {
                val score = detections[4 + j][i]
                if (score > maxScore) {
                    maxScore = score
                    classId = j
                }
            }

            if (maxScore > confidenceThreshold) {
                val maskCoeffs = FloatArray(numCoefficients) { k -> detections[4 + numClasses + k][i] }
                detectedObjects.add(Detection(x, y, w, h, maxScore, classId, maskCoeffs))
            }
        }

        val finalDetections = applyNMS(detectedObjects, iouThreshold)
        return drawMasksOnly(originalBitmap, finalDetections, masks)
    }

    private data class Detection(
        val x: Float, val y: Float, val width: Float, val height: Float,
        val confidence: Float, val classId: Int, val maskCoefficients: FloatArray
    )

    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        val suppressed = mutableSetOf<Int>()

        for (i in sorted.indices) {
            if (i in suppressed) continue
            selected.add(sorted[i])
            for (j in (i + 1) until sorted.size) {
                if (j !in suppressed && calculateIoU(sorted[i], sorted[j]) > iouThreshold) {
                    suppressed.add(j)
                }
            }
        }
        return selected
    }

    private fun calculateIoU(d1: Detection, d2: Detection): Float {
        val x1Min = d1.x - d1.width / 2; val y1Min = d1.y - d1.height / 2
        val x1Max = d1.x + d1.width / 2; val y1Max = d1.y + d1.height / 2
        val x2Min = d2.x - d2.width / 2; val y2Min = d2.y - d2.height / 2
        val x2Max = d2.x + d2.width / 2; val y2Max = d2.y + d2.height / 2

        val intersectArea = max(0f, min(x1Max, x2Max) - max(x1Min, x2Min)) *
                max(0f, min(y1Max, y2Max) - max(y1Min, y2Min))
        val unionArea = d1.width * d1.height + d2.width * d2.height - intersectArea
        return if (unionArea > 0) intersectArea / unionArea else 0f
    }

    private fun drawMasksOnly(
        bitmap: Bitmap,
        detections: List<Detection>,
        masks: Array<Array<FloatArray>>
    ): Bitmap {
        val segmentedBitmap = ensureSoftwareArgb(bitmap).copy(Bitmap.Config.ARGB_8888, true)
        val maskOnlyBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)

        val segmentedCanvas = Canvas(segmentedBitmap)
        val maskOnlyCanvas = Canvas(maskOnlyBitmap)

        maskOnlyCanvas.drawColor(Color.BLACK)

        val classMasks = mutableMapOf<Int, BooleanArray>()
        val imageSize = bitmap.width * bitmap.height
        for (classId in 0 until classLabels.size) {
            classMasks[classId] = BooleanArray(imageSize)
        }

        for (detection in detections) {
            drawMask(segmentedCanvas, maskOnlyCanvas, masks, detection, bitmap, classMasks)
        }

        val totalImagePixels = imageSize
        val classStatsMap = mutableMapOf<String, ClassStats>()
        var totalSegmentedPixels = 0

        // Count objects per class from detections
        val objectCounts = mutableMapOf<Int, Int>()
        for (detection in detections) {
            objectCounts[detection.classId] = objectCounts.getOrDefault(detection.classId, 0) + 1
        }

        for (classId in 0 until classLabels.size) {
            val uniquePixels = classMasks[classId]!!.count { it }
            if (uniquePixels > 0) {
                val coverage = (uniquePixels.toFloat() / totalImagePixels) * 100
                val objectCount = objectCounts.getOrDefault(classId, 0)
                classStatsMap[classLabels[classId]] = ClassStats(
                    className = classLabels[classId],
                    pixels = uniquePixels,
                    coverage = coverage,
                    color = classColors[classId] ?: Color.GRAY,
                    objectCount = objectCount
                )
                totalSegmentedPixels += uniquePixels
            }
        }

        val imageCoverage = (totalSegmentedPixels.toFloat() / totalImagePixels) * 100
        val delegateName = getCurrentDelegate()

        lastStats = SegmentationStats(
            classStats = classStatsMap,
            totalSegmentedPixels = totalSegmentedPixels,
            totalImagePixels = totalImagePixels,
            imageCoverage = imageCoverage,
            processingTimeMs = lastStats?.processingTimeMs ?: 0L,
            preprocessTimeMs = lastStats?.preprocessTimeMs ?: 0L,
            inferenceTimeMs = lastStats?.inferenceTimeMs ?: 0L,
            postprocessTimeMs = lastStats?.postprocessTimeMs ?: 0L,
            delegateUsed = delegateName
        )

        try { lastSegmentedBitmap?.recycle() } catch (_: Exception) {}
        try { lastMaskOnlyBitmap?.recycle() } catch (_: Exception) {}

        lastSegmentedBitmap = segmentedBitmap
        lastMaskOnlyBitmap = maskOnlyBitmap

        Log.d(TAG, "Stats: Total=${totalSegmentedPixels} px, Coverage=${"%.2f".format(imageCoverage)}%")
        return segmentedBitmap
    }

    private fun drawMask(
        segmentedCanvas: Canvas,
        maskOnlyCanvas: Canvas,
        masks: Array<Array<FloatArray>>,
        detection: Detection,
        bitmap: Bitmap,
        classMasks: MutableMap<Int, BooleanArray>
    ) {
        val prototypeWidth = masks[0].size
        val prototypeHeight = masks.size
        val combinedMask = FloatArray(prototypeWidth * prototypeHeight)

        for (y in 0 until prototypeHeight) {
            for (x in 0 until prototypeWidth) {
                var sum = 0f
                for (k in detection.maskCoefficients.indices) {
                    sum += detection.maskCoefficients[k] * masks[y][x][k]
                }
                combinedMask[y * prototypeWidth + x] = sum
            }
        }

        val maskColor = classColors[detection.classId] ?: Color.YELLOW
        val maskPixelsTransparent = IntArray(prototypeWidth * prototypeHeight)
        val maskPixelsOpaque = IntArray(prototypeWidth * prototypeHeight)

        for (i in combinedMask.indices) {
            val sigmoid = 1.0f / (1.0f + Math.exp(-combinedMask[i].toDouble())).toFloat()
            if (sigmoid > 0.5f) {
                maskPixelsTransparent[i] = Color.argb(120, Color.red(maskColor), Color.green(maskColor), Color.blue(maskColor))
                maskPixelsOpaque[i] = Color.argb(255, Color.red(maskColor), Color.green(maskColor), Color.blue(maskColor))
            } else {
                maskPixelsTransparent[i] = Color.TRANSPARENT
                maskPixelsOpaque[i] = Color.TRANSPARENT
            }
        }

        val maskBitmapTransparent = Bitmap.createBitmap(maskPixelsTransparent, prototypeWidth, prototypeHeight, Bitmap.Config.ARGB_8888)
        val maskBitmapOpaque = Bitmap.createBitmap(maskPixelsOpaque, prototypeWidth, prototypeHeight, Bitmap.Config.ARGB_8888)

        val mask640Transparent = Bitmap.createScaledBitmap(maskBitmapTransparent, INPUT_SIZE, INPUT_SIZE, true)
        val mask640Opaque = Bitmap.createScaledBitmap(maskBitmapOpaque, INPUT_SIZE, INPUT_SIZE, true)

        val boxLeft = (detection.x - detection.width / 2).coerceIn(0f, INPUT_SIZE - 1f)
        val boxTop = (detection.y - detection.height / 2).coerceIn(0f, INPUT_SIZE - 1f)
        val boxRight = (detection.x + detection.width / 2).coerceIn(boxLeft + 1f, INPUT_SIZE.toFloat())
        val boxBottom = (detection.y + detection.height / 2).coerceIn(boxTop + 1f, INPUT_SIZE.toFloat())

        val boxWidth = (boxRight - boxLeft).toInt().coerceAtLeast(1)
        val boxHeight = (boxBottom - boxTop).toInt().coerceAtLeast(1)

        try {
            val croppedMaskTransparent = Bitmap.createBitmap(mask640Transparent, boxLeft.toInt(), boxTop.toInt(), boxWidth, boxHeight)
            val croppedMaskOpaque = Bitmap.createBitmap(mask640Opaque, boxLeft.toInt(), boxTop.toInt(), boxWidth, boxHeight)

            val scaleToUse = if (scaleX > 0.001f) scaleX else 1.0f
            val origLeft = ((boxLeft - padLeft) / scaleToUse).coerceIn(0f, bitmap.width.toFloat()).toInt()
            val origTop = ((boxTop - padTop) / scaleToUse).coerceIn(0f, bitmap.height.toFloat()).toInt()
            val origWidth = (boxWidth / scaleToUse).coerceAtLeast(1f).coerceAtMost(bitmap.width.toFloat()).toInt()
            val origHeight = (boxHeight / scaleToUse).coerceAtLeast(1f).coerceAtMost(bitmap.height.toFloat()).toInt()

            val finalMaskTransparent = Bitmap.createScaledBitmap(croppedMaskTransparent, origWidth, origHeight, true)
            val finalMaskOpaque = Bitmap.createScaledBitmap(croppedMaskOpaque, origWidth, origHeight, true)

            val maskArray = classMasks[detection.classId]!!
            val pixels = IntArray(finalMaskTransparent.width * finalMaskTransparent.height)
            finalMaskTransparent.getPixels(pixels, 0, finalMaskTransparent.width, 0, 0, finalMaskTransparent.width, finalMaskTransparent.height)

            for (y in 0 until finalMaskTransparent.height) {
                for (x in 0 until finalMaskTransparent.width) {
                    val imgX = origLeft + x
                    val imgY = origTop + y
                    if (imgX < bitmap.width && imgY < bitmap.height) {
                        if (pixels[y * finalMaskTransparent.width + x] != Color.TRANSPARENT) {
                            val flatIndex = imgY * bitmap.width + imgX
                            if (flatIndex < maskArray.size) {
                                maskArray[flatIndex] = true
                            }
                        }
                    }
                }
            }

            segmentedCanvas.drawBitmap(finalMaskTransparent, origLeft.toFloat(), origTop.toFloat(), null)
            maskOnlyCanvas.drawBitmap(finalMaskOpaque, origLeft.toFloat(), origTop.toFloat(), null)

            try { croppedMaskTransparent.recycle() } catch (_: Exception) {}
            try { croppedMaskOpaque.recycle() } catch (_: Exception) {}
            try { finalMaskTransparent.recycle() } catch (_: Exception) {}
            try { finalMaskOpaque.recycle() } catch (_: Exception) {}

        } catch (e: Exception) {
            Log.e(TAG, "Failed to draw mask: ${e.message}")
        } finally {
            try { maskBitmapTransparent.recycle() } catch (_: Exception) {}
            try { maskBitmapOpaque.recycle() } catch (_: Exception) {}
            try { mask640Transparent.recycle() } catch (_: Exception) {}
            try { mask640Opaque.recycle() } catch (_: Exception) {}
        }
    }

    private fun extractSkelsFromAssetsIfPresent() {
        try {
            val assetDir = "qnn_skel"
            val files = context.assets.list(assetDir) ?: return
            if (files.isEmpty()) return
            val outDir = java.io.File(context.filesDir, assetDir)
            if (!outDir.exists()) outDir.mkdirs()
            for (name in files) {
                val inputPath = "$assetDir/$name"
                context.assets.open(inputPath).use { input ->
                    val outFile = java.io.File(outDir, name)
                    context.openFileOutput("$assetDir/$name", Context.MODE_PRIVATE).use { /* not used */ }
                    java.io.FileOutputStream(outFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            Log.d(TAG, "Extracted ${files.size} skel files to ${outDir.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "extractSkelsFromAssetsIfPresent failed: ${e.message}")
        }
    }

    private fun warmupInterpreter(name: String, interpreter: Interpreter?) {
        if (interpreter == null) return
        val input = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE * FLOAT_SIZE)
        input.order(ByteOrder.nativeOrder())
        input.rewind()

        val detectionsOutput = Array(OUTPUT_0_DIMS[0]) {
            Array(OUTPUT_0_DIMS[1]) { FloatArray(OUTPUT_0_DIMS[2]) }
        }
        val masksOutput = Array(OUTPUT_1_DIMS[0]) {
            Array(OUTPUT_1_DIMS[1]) {
                Array(OUTPUT_1_DIMS[2]) { FloatArray(OUTPUT_1_DIMS[3]) }
            }
        }
        val outMap = TreeMap<Int, Any>().apply {
            put(0, detectionsOutput)
            put(1, masksOutput)
        }
        // 2 quick runs
        repeat(2) {
            try { interpreter.runForMultipleInputsOutputs(arrayOf(input), outMap) } catch (_: Exception) {}
        }
        Log.d(TAG, "Warmup completed on $name")
    }

    fun close() {
        Log.d(TAG, "Closing SegmentationHelper...")

        try { interpreterGPU?.close() } catch (e: Exception) {
            Log.e(TAG, "Error closing GPU interpreter", e)
        }

        try { interpreterNNAPI?.close() } catch (e: Exception) {
            Log.e(TAG, "Error closing NNAPI interpreter", e)
        }

        try { interpreterCPU?.close() } catch (e: Exception) {
            Log.e(TAG, "Error closing CPU interpreter", e)
        }

        try { gpuDelegate?.close() } catch (e: Exception) {
            Log.e(TAG, "Error closing TFLite GPU delegate", e)
        }
        
        try { nnapiDelegate?.close() } catch (e: Exception) {
            Log.e(TAG, "Error closing NNAPI delegate", e)
        }

        interpreterGPU = null
        interpreterNNAPI = null
        interpreterCPU = null
        gpuDelegate = null
        nnapiDelegate = null

        try { lastMaskOnlyBitmap?.recycle() } catch (_: Exception) {}
        try { lastSegmentedBitmap?.recycle() } catch (_: Exception) {}

        Log.d(TAG, "SegmentationHelper closed successfully")
    }
}