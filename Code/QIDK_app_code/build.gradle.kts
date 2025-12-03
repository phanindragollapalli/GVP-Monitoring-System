// Delete the entire 'buildscript' block.
// Use the 'plugins' block to manage all your plugins.
plugins {
    // Reference versions from your libs.versions.toml file
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("com.google.gms.google-services") version "4.4.2" apply false

    // You are using Compose, so you need the Compose plugin.
    // It's in your libs.versions.toml but was missing here.
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}

// It is safe to remove the 'buildscript' block entirely.

tasks.register("clean", Delete::class) {
    delete(rootProject.layout.buildDirectory)
}
