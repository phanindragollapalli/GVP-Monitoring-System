package com.example.esw

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.*
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import android.os.Environment
import com.google.firebase.FirebaseApp
import com.google.firebase.database.FirebaseDatabase
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import org.json.JSONObject
import com.google.firebase.storage.FirebaseStorage

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 100
    }

    // ---------- UI ----------
    private lateinit var imageView: ImageView
    private lateinit var statsTextView: TextView
    private lateinit var totalPixelsLabel: TextView
    private lateinit var coverageLabel: TextView
    private lateinit var processingTimeLabel: TextView
    private lateinit var preprocessingTimeLabel: TextView
    private lateinit var inferenceTimeLabel: TextView
    private lateinit var postprocessingTimeLabel: TextView
    private lateinit var toggleViewButton: Button
    private lateinit var autoClickButton: Button
    private lateinit var autoClickStatusText: TextView
    private lateinit var switchDelegateButton: Button

    private var isCapturing = false

    // ---------- Camera ----------
    private var previewView: PreviewView? = null
    private var imageCapture: ImageCapture? = null
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null

    // ---------- Segmentation ----------
    private lateinit var segHelper: SegmentationHelper
    private var originalBitmap: Bitmap? = null
    private var segmentedBitmap: Bitmap? = null
    private var isSegmentedView = false

    // ---------- Auto-capture ----------
    private var isAutoCaptureRunning = false
    private var autoCaptureIntervalMs = 5000L
    private val handler = Handler(Looper.getMainLooper())
    private var autoCaptureCount = 0

    // ---------- Firebase ----------
    private lateinit var firebaseDb: FirebaseDatabase

    // -------------------------------------------------------------------------
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // ---- Firebase ----
        FirebaseApp.initializeApp(this)
        firebaseDb = FirebaseDatabase.getInstance(
            "https://gvp-segmentation-default-rtdb.asia-southeast1.firebasedatabase.app"
        )
        // ------------------

        initViews()
        segHelper = SegmentationHelper(this)

        checkAndRequestPermissions()
        setupButtons()
        updateDelegateButtonText()

        Log.d(TAG, "MainActivity created successfully")
    }

    // -------------------------------------------------------------------------
    private fun initViews() {
        imageView = findViewById(R.id.imageView)
        statsTextView = findViewById(R.id.statsTextView)
        totalPixelsLabel = findViewById(R.id.totalPixelsLabel)
        coverageLabel = findViewById(R.id.coverageLabel)
        processingTimeLabel = findViewById(R.id.processingTimeLabel)
        preprocessingTimeLabel = findViewById(R.id.preprocessingTimeLabel)
        inferenceTimeLabel = findViewById(R.id.inferenceTimeLabel)
        postprocessingTimeLabel = findViewById(R.id.postprocessingTimeLabel)
        toggleViewButton = findViewById(R.id.btnToggleView)
        autoClickButton = findViewById(R.id.btnAutoClick)
        autoClickStatusText = findViewById(R.id.autoClickStatusText)
        previewView = findViewById(R.id.previewView)
        switchDelegateButton = findViewById(R.id.btnSwitchDelegate)
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnCapture).setOnClickListener {
            takePhotoAndSegment(saveToGallery = false)
        }

        findViewById<Button>(R.id.btnSelect).setOnClickListener {
            openGallery()
        }

        toggleViewButton.setOnClickListener {
            toggleView()
        }

        autoClickButton.setOnClickListener {
            if (isAutoCaptureRunning) stopAutoCapture() else showAutoCaptureDialog()
        }

        // Single tap now opens the delegate picker directly
        switchDelegateButton.setOnClickListener {
            showDelegatePicker()
        }
        // Long-press cycles (optional)
        switchDelegateButton.setOnLongClickListener {
            segHelper.switchDelegate()
            updateDelegateButtonText()
            val current = segHelper.getCurrentDelegate()
            Toast.makeText(this, "Switched to: $current", Toast.LENGTH_SHORT).show()
            Log.d(TAG, "User switched to: $current (long-press cycle)")
            true
        }
    }

    private fun updateDelegateButtonText() {
        val current = segHelper.getCurrentDelegate()
        val gpuAvailable = segHelper.isGPUAvailable()
        val nnapiAvailable = segHelper.isNNAPIAvailable()
        val cpuAvailable = segHelper.isCPUAvailable()

        val totalAvailable = listOf(gpuAvailable, nnapiAvailable, cpuAvailable).count { it }

        val status = if (totalAvailable == 0) {
            "ERROR: No processors"
        } else {
            "Current: $current"
        }

        switchDelegateButton.text = status
        switchDelegateButton.isEnabled = totalAvailable >= 2

        Log.d(TAG, "Delegate status: $status (GPU: $gpuAvailable, NNAPI: $nnapiAvailable, CPU: $cpuAvailable)")
    }

    // -------------------------------------------------------------------------
    private fun checkAndRequestPermissions() {
        val needed = mutableListOf<String>()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            needed.add(Manifest.permission.CAMERA)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED) {
                needed.add(Manifest.permission.READ_MEDIA_IMAGES)
            }
        }

        if (needed.isNotEmpty()) {
            Log.d(TAG, "Requesting permissions: ${needed.joinToString()}")
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSION_REQUEST_CODE)
        } else {
            Log.d(TAG, "All permissions granted")
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                Log.d(TAG, "Permissions granted")
                startCamera()
            } else {
                Log.e(TAG, "Permissions denied")
                Toast.makeText(
                    this,
                    "Permissions denied – app cannot work",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    // -------------------------------------------------------------------------
    private fun startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture?.addListener({
            try {
                val cameraProvider = cameraProviderFuture?.get() ?: return@addListener

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView?.surfaceProvider)
                }

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setFlashMode(ImageCapture.FLASH_MODE_OFF)
                    .setTargetRotation(previewView?.display?.rotation ?: Surface.ROTATION_0)
                    .build()

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this as LifecycleOwner,
                    cameraSelector,
                    preview,
                    imageCapture
                )

                // Ensure preview is visible
                previewView?.visibility = View.VISIBLE
                imageView.visibility = View.GONE

                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhotoAndSegment(saveToGallery: Boolean) {
        val ic = imageCapture
        if (ic == null) {
            Log.w(TAG, "ImageCapture not initialized – restarting camera")
            runOnUiThread { Toast.makeText(this, "Starting camera…", Toast.LENGTH_SHORT).show() }
            startCamera()
            return
        }
        if (isCapturing) {
            Log.d(TAG, "Capture already in flight, ignoring tap")
            return
        }
        isCapturing = true

        // Ensure user sees the preview before capture
        previewView?.visibility = View.VISIBLE
        imageView.visibility = View.GONE

        // Capture to a temporary JPEG file to avoid YUV->RGB conversion issues
        val photoFile = java.io.File(cacheDir, "capt_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        ic.takePicture(outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                try {
                    val bmp = decodeJpegUpright(photoFile.absolutePath)
                    val src = if (isAutoCaptureRunning || saveToGallery) "auto" else "capture"
                    runOnUiThread { processBitmap(bmp, saveToGallery, src) }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to decode captured image", e)
                    runOnUiThread { Toast.makeText(this@MainActivity, "Decode failed", Toast.LENGTH_SHORT).show() }
                } finally {
                    isCapturing = false
                    try { photoFile.delete() } catch (_: Exception) {}
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "Photo capture failed", exception)
                isCapturing = false
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Capture failed: ${exception.message}", Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                try {
                    val bmp = MediaStore.Images.Media.getBitmap(contentResolver, it)
                    processBitmap(bmp, saveToGallery = false, source = "gallery")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load image from gallery", e)
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                }
            }
        }

    private fun openGallery() = galleryLauncher.launch("image/*")

    private fun decodeJpegUpright(path: String): Bitmap {
        val raw = BitmapFactory.decodeFile(path)
            ?: throw IllegalStateException("Failed to decode: $path")
        val bmp = raw.copy(Bitmap.Config.ARGB_8888, true)
        if (bmp !== raw) try { raw.recycle() } catch (_: Exception) {}

        return try {
            val exif = ExifInterface(path)
            val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            val m = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
                ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
                ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
                else -> {}
            }
            if (!m.isIdentity) {
                val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                if (rotated !== bmp) try { bmp.recycle() } catch (_: Exception) {}
                rotated
            } else bmp
        } catch (_: Exception) {
            bmp
        }
    }

    // -------------------------------------------------------------------------
    private fun processBitmap(bitmap: Bitmap, saveToGallery: Boolean, source: String) {
        originalBitmap = bitmap

        // Run segmentation off the UI thread so subsequent captures are responsive
        inferenceExecutor.execute {
            val result = segHelper.runSegmentation(bitmap)
            val stats = segHelper.getLastStats()

            runOnUiThread {
                segmentedBitmap = result
                imageView.setImageBitmap(segmentedBitmap)
                previewView?.visibility = View.GONE
                imageView.visibility = View.VISIBLE

                totalPixelsLabel.text = "Total pixels: ${bitmap.width * bitmap.height}"
                if (stats != null) {
                    coverageLabel.text = "Coverage: ${"%.2f".format(stats.imageCoverage)} %"
                    processingTimeLabel.text = "${stats.delegateUsed} Total: ${stats.processingTimeMs} ms"
                    preprocessingTimeLabel.text = "Preprocess: ${stats.preprocessTimeMs} ms"
                    inferenceTimeLabel.text = "Inference: ${stats.inferenceTimeMs} ms"
                    postprocessingTimeLabel.text = "Post-process: ${stats.postprocessTimeMs} ms"

                    val classLines = stats.classStats.map { (name, cs) ->
                        when (name) {
                            "Person", "Animal" -> "$name: ${cs.objectCount} detected"
                            else -> "$name: ${cs.pixels} px (${"%.2f".format(cs.coverage)} %)"
                        }
                    }.joinToString("\n")
                    statsTextView.text = if (classLines.isEmpty()) "No objects detected" else classLines
                } else {
                    coverageLabel.text = "Coverage: -"
                    processingTimeLabel.text = "Processing: -"
                    preprocessingTimeLabel.text = "Preprocess: -"
                    inferenceTimeLabel.text = "Inference: -"
                    postprocessingTimeLabel.text = "Post-process: -"
                    statsTextView.text = "Error during inference"
                }

                // Always send metadata to Firebase after processing
                pushMetadataToFirebase(stats, source)

                if (saveToGallery) {
                    saveSegmentedToGallery()
                }
            }
        }
    }

    private fun toggleView() {
        if (isSegmentedView) {
            imageView.setImageBitmap(originalBitmap)
            toggleViewButton.text = "Show Segmented"
        } else {
            imageView.setImageBitmap(segmentedBitmap)
            toggleViewButton.text = "Show Original"
        }
        isSegmentedView = !isSegmentedView
    }

    // -------------------------------------------------------------------------
    private fun pushMetadataToFirebase(stats: SegmentationHelper.SegmentationStats?, source: String) {
        val dbRef = firebaseDb.getReference("segmentation_results")

        val ts = System.currentTimeMillis()
        val data = mutableMapOf<String, Any>(
            "timestamp"          to ts,
            "timestampIST"       to formatToIST(ts),
            "captureCount"       to autoCaptureCount,
            "totalPixels"        to (originalBitmap?.width?.times(originalBitmap?.height ?: 0) ?: 0),
            "coverage"           to (stats?.imageCoverage ?: 0f),
            "processingTimeMs"   to (stats?.processingTimeMs ?: 0L),
            "delegateUsed"       to (stats?.delegateUsed ?: "Unknown"),
            "classBreakdown"     to (stats?.classStats?.mapValues { it.value.coverage } ?: emptyMap()),
            "source"             to source
        )

        val bmp = segmentedBitmap

        // Always write a local copy of metadata + image on device storage
        saveResultLocally(ts, data, bmp)

        if (bmp != null) {
            try {
                val path = "segmented/seg_${ts}.jpg"
                val storageRef = FirebaseStorage.getInstance().reference.child(path)
                val baos = ByteArrayOutputStream()
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, baos)
                val bytes = baos.toByteArray()
                storageRef.putBytes(bytes)
                    .addOnSuccessListener {
                        storageRef.downloadUrl
                            .addOnSuccessListener { uri ->
                                data["imageUrl"] = uri.toString()
                                dbRef.push().setValue(data)
                                    .addOnSuccessListener { if (!isAutoCaptureRunning) Toast.makeText(this, "Data uploaded to Firebase", Toast.LENGTH_SHORT).show() }
                                    .addOnFailureListener { e -> Log.e(TAG, "Failed to upload data to Firebase", e) }
                            }
                            .addOnFailureListener { e ->
                                Log.w(TAG, "Get downloadUrl failed, writing data without imageUrl", e)
                                dbRef.push().setValue(data)
                            }
                    }
                    .addOnFailureListener { e ->
                        Log.w(TAG, "Image upload failed, writing data without imageUrl", e)
                        dbRef.push().setValue(data)
                    }
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error uploading image", e)
                dbRef.push().setValue(data)
            }
        } else {
            dbRef.push().setValue(data)
                .addOnSuccessListener {
                    Log.d(TAG, "Data uploaded to Firebase successfully")
                    if (!isAutoCaptureRunning) {
                        Toast.makeText(this, "Data uploaded to Firebase", Toast.LENGTH_SHORT).show()
                    }
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Failed to upload data to Firebase", e)
                    if (!isAutoCaptureRunning) {
                        Toast.makeText(this, "Failed to upload data", Toast.LENGTH_SHORT).show()
                    }
                }
        }
    }

    private fun saveResultLocally(
        timestamp: Long,
        data: MutableMap<String, Any>,
        bmp: Bitmap?
    ) {
        try {
            val rootDir = File(getExternalFilesDir(null), "segmentation_captures")
            if (!rootDir.exists()) {
                rootDir.mkdirs()
            }

            var localImagePath = ""

            // Save image if available
            if (bmp != null) {
                val imgFile = File(rootDir, "seg_${timestamp}.jpg")
                FileOutputStream(imgFile).use { out ->
                    bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }
                localImagePath = imgFile.absolutePath
                data["localImagePath"] = localImagePath
            }

            // Prepare a single CSV file with one row per capture
            val csvFile = File(rootDir, "captures.csv")
            val needsHeader = !csvFile.exists()

            FileWriter(csvFile, true).use { writer ->
                if (needsHeader) {
                    writer.appendLine("timestamp,timestampIST,captureCount,totalPixels,coverage,processingTimeMs,delegateUsed,source,localImagePath")
                }

                val timestampIST = data["timestampIST"]?.toString() ?: ""
                val captureCount = data["captureCount"]?.toString() ?: ""
                val totalPixels = data["totalPixels"]?.toString() ?: ""
                val coverage = data["coverage"]?.toString() ?: ""
                val processingTimeMs = data["processingTimeMs"]?.toString() ?: ""
                val delegateUsed = data["delegateUsed"]?.toString() ?: ""
                val source = data["source"]?.toString() ?: ""

                // Basic CSV escaping for fields that might contain commas
                fun esc(value: String): String = if (value.contains(",")) "\"${value.replace("\"", "\"\"")}\"" else value

                val line = listOf(
                    timestamp.toString(),
                    esc(timestampIST),
                    captureCount,
                    totalPixels,
                    coverage,
                    processingTimeMs,
                    esc(delegateUsed),
                    esc(source),
                    esc(localImagePath)
                ).joinToString(",")

                writer.appendLine(line)
            }

            Log.d(TAG, "Appended local result to ${csvFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save result locally", e)
        }
    }

    private fun formatToIST(millis: Long): String {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss z", Locale.US)
        sdf.timeZone = TimeZone.getTimeZone("Asia/Kolkata")
        return sdf.format(Date(millis))
    }

    private fun saveSegmentedToGallery() {
        val bmp = segmentedBitmap ?: return
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val filename = "seg_$timestamp.jpg"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/ESWSeg")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        uri?.let {
            try {
                contentResolver.openOutputStream(it)?.use { out ->
                    bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(it, values, null, null)

                if (!isAutoCaptureRunning) {
                    Toast.makeText(this, "Saved $filename", Toast.LENGTH_SHORT).show()
                }
                Log.d(TAG, "Image saved: $filename")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save image", e)
                Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // -------------------------------------------------------------------------
    private fun showDelegatePicker() {
        // If GPU is unavailable, show quick reason to the user
        if (!segHelper.isGPUAvailable()) {
            segHelper.getGpuUnavailableReason()?.let {
                Toast.makeText(this, "GPU unavailable: $it", Toast.LENGTH_LONG).show()
            }
        }

        val items = mutableListOf<String>()
        val types = mutableListOf<SegmentationHelper.DelegateType>()
        if (segHelper.isGPUAvailable()) { items += "GPU (TFLite)"; types += SegmentationHelper.DelegateType.GPU }
        if (segHelper.isNNAPIAvailable()) { items += "NNAPI"; types += SegmentationHelper.DelegateType.NNAPI }
        if (segHelper.isCPUAvailable()) { items += "CPU"; types += SegmentationHelper.DelegateType.CPU }
        if (items.isEmpty()) {
            Toast.makeText(this, "No processors available", Toast.LENGTH_SHORT).show(); return
        }
        AlertDialog.Builder(this)
            .setTitle("Select Processor")
            .setSingleChoiceItems(items.toTypedArray(), -1) { dialog, which ->
                val ok = segHelper.forceDelegate(types[which])
                updateDelegateButtonText()
                val current = segHelper.getCurrentDelegate()
                Toast.makeText(this, if (ok) "Using: $current" else "Not available", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showAutoCaptureDialog() {
        val view = layoutInflater.inflate(R.layout.dialog_auto_click, null)
        val input = view.findViewById<EditText>(R.id.intervalInput)

        AlertDialog.Builder(this)
            .setView(view)
            .setPositiveButton("Start") { _, _ ->
                val secs = input.text.toString().toLongOrNull() ?: 5
                autoCaptureIntervalMs = secs.coerceAtLeast(1) * 1000L
                startAutoCapture()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startAutoCapture() {
        isAutoCaptureRunning = true
        autoCaptureCount = 0
        autoClickButton.text = "Stop Auto Capture"
        autoClickStatusText.text = "Running… (0)"
        previewView?.visibility = View.VISIBLE
        imageView.visibility = View.GONE
        // Write a heartbeat so we can verify DB connectivity from console
        try {
            firebaseDb.getReference("app_status/auto_capture_started_at").setValue(System.currentTimeMillis())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write heartbeat to Firebase", e)
        }
        startCamera()
        scheduleNextCapture()
        Log.d(TAG, "Auto-capture started (interval: ${autoCaptureIntervalMs}ms)")
    }

    private fun stopAutoCapture() {
        isAutoCaptureRunning = false
        handler.removeCallbacksAndMessages(null)
        autoClickButton.text = "Start Auto Capture"
        autoClickStatusText.text = "Stopped (Total: $autoCaptureCount)"
        Log.d(TAG, "Auto-capture stopped (total captures: $autoCaptureCount)")
    }

    private fun scheduleNextCapture() {
        if (!isAutoCaptureRunning) return

        handler.postDelayed({
            takePhotoAndSegment(saveToGallery = true)
            autoCaptureCount++
            autoClickStatusText.text = "Running… ($autoCaptureCount)"
            scheduleNextCapture()
        }, autoCaptureIntervalMs)
    }

    // -------------------------------------------------------------------------
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "MainActivity destroying...")
        segHelper.close()
        cameraExecutor.shutdown()
        inferenceExecutor.shutdown()
        handler.removeCallbacksAndMessages(null)
    }
}