# Step-by-Step Guide: Building Your Android & Wear OS App 🤖⌚

Since this is your first time, we will build this step-by-step using **Android Studio**. We are creating two apps that talk to each other:
1.  **Wear OS App**: Runs on the watch, reads heart rate, sends it to phone.
2.  **Mobile App**: Runs on the phone, receives data, sends it to your Web Dashboard.

---

## Phase 1: Setup & Project Creation 🛠️

1.  **Open Android Studio**.
2.  Click **New Project**.
3.  **Choose Project Template**:
    *   Select **"Wear OS"** tab on the left.
    *   Choose **"Empty Serve OS App"** (or similar Empty Activity for Wear).
    *   Click **Next**.
4.  **Configure Project**:
    *   **Name**: `AntigravityWear`
    *   **Package Name**: `com.antigravity.wear`
    *   **Language**: **Kotlin**
    *   **Minimum SDK**: API 30 (Android 11) is a good safe bet for Galaxy Watch 4/5/6.
    *   Click **Finish**.
    *   *Wait for Gradle to finish syncing (bottom status bar).*

5.  **Add the Mobile Module**:
    *   Go to **File > New > New Module**.
    *   Select **"Phone & Tablet"**.
    *   Choose **"Empty Activity"**.
    *   **Module Name**: `mobile`
    *   **Package Name**: `com.antigravity.mobile`
    *   Click **Finish**.

Now you have one project with two modules: `app` (Watch) and `mobile` (Phone).

---

## Phase 2: The Wear OS App (Sender) ⌚

We need to tell the watch to use its sensors and Bluetooth.

### Step 2.1: Add Permissions
Open `app/src/main/AndroidManifest.xml` (in the Wear module) and add these **above** the `<application>` tag:

```xml
<uses-permission android:name="android.permission.BODY_SENSORS" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

### Step 2.2: Add Dependencies
Open `build.gradle.kts (Module :app)` and add these to `dependencies`:

```kotlin
implementation("com.google.android.gms:play-services-wearable:18.1.0")
```
*Click **Sync Now** at the top right.*

### Step 2.3: Write the Code
Open `MainActivity.kt` in the Wear module. Replace the content with this simple logic:

```kotlin
package com.antigravity.wear

import android.app.Activity
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import com.google.android.gms.wearable.PutDataMapRequest
import com.google.android.gms.wearable.Wearable

class MainActivity : Activity(), SensorEventListener {
    private lateinit var sensorManager: SensorManager
    private var heartRateSensor: Sensor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main) // Ensure you have a layout or remove this if using Compose

        // 1. Get Sensor Manager
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        
        // 2. Request Permissions (Run-time permission logic omitted for brevity, ensure you grant it in Settings manually for testing!)
    }

    override fun onResume() {
        super.onResume()
        // 3. Start Listening
        heartRateSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    // 4. Handle Sensor Data
    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_HEART_RATE) {
            val heartRate = event.values[0]
            Log.d("WearApp", "Heart Rate: $heartRate")
            sendDataToPhone(heartRate)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // 5. Send to Phone (Data Layer API)
    private fun sendDataToPhone(hr: Float) {
        val putDataMapReq = PutDataMapRequest.create("/heart_rate")
        putDataMapReq.dataMap.putFloat("bpm", hr)
        putDataMapReq.dataMap.putLong("timestamp", System.currentTimeMillis())
        val putDataReq = putDataMapReq.asPutDataRequest().setUrgent()
        Wearable.getDataClient(this).putDataItem(putDataReq)
    }
}
```

---

## Phase 3: The Mobile App (Receiver) 📱

The phone needs to listen to the watch and forward to the PC.

### Step 3.1: Add Dependencies
Open `build.gradle.kts (Module :mobile)`:

```kotlin
implementation("com.google.android.gms:play-services-wearable:18.1.0")
implementation("com.squareup.okhttp3:okhttp:4.10.0")
```
*Click **Sync Now**.*

### Step 3.2: Write the Code
Open `MainActivity.kt` in the Mobile module:

```kotlin
package com.antigravity.mobile

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.wearable.DataClient
import com.google.android.gms.wearable.DataEvent
import com.google.android.gms.wearable.DataEventBuffer
import com.google.android.gms.wearable.DataMapItem
import com.google.android.gms.wearable.Wearable
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener

class MainActivity : AppCompatActivity(), DataClient.OnDataChangedListener {

    private var webSocket: WebSocket? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 1. Connect to Web Dashboard (Replace with your PC's IP!)
        connectWebSocket("ws://192.168.1.X:8080") 
    }

    override fun onResume() {
        super.onResume()
        // 2. Listen to Watch
        Wearable.getDataClient(this).addListener(this)
    }

    override fun onPause() {
        super.onPause()
        Wearable.getDataClient(this).removeListener(this)
    }

    // 3. Handle data from Watch
    override fun onDataChanged(dataEvents: DataEventBuffer) {
        for (event in dataEvents) {
            if (event.type == DataEvent.TYPE_CHANGED) {
                val path = event.dataItem.uri.path
                if (path == "/heart_rate") {
                    val dataMap = DataMapItem.fromDataItem(event.dataItem).dataMap
                    val hr = dataMap.getFloat("bpm")
                    Log.d("MobileApp", "Received HR: $hr")
                    
                    // 4. Forward to Web
                    sendToWeb(hr)
                }
            }
        }
    }

    private fun connectWebSocket(url: String) {
        val client = OkHttpClient()
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                Log.d("Web", "Connected!")
            }
        })
    }

    private fun sendToWeb(hr: Float) {
        val json = """{"heartRate": $hr}"""
        webSocket?.send(json)
    }
}
```

---

## Phase 4: Running It 🚀

1.  **Start your Web App**: Run `npm run dev` and ensure the WebSocket Mock Server or your real Backend is running.
2.  **Run Mobile App**: Connect your Android Phone via USB, select "mobile" configuration in Android Studio, and click Run.
3.  **Run Wear App**: Connect your Samsung Watch via WiFi debugging (or Bluetooth), select "app" configuration, and click Run.
4.  **Test**: Open the app on the watch. It should log heart rate. Check your phone logs (Logcat) to see it receiving data, and check your Web Dashboard!
