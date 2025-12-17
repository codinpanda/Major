# ⌚ Ultimate Guide: Connecting Samsung Watch to Web Dashboard

**Goal**: Stream Heart Rate from **Galaxy Watch 4/5/6** -> **Android Phone** -> **Web Dashboard** (Real-time).

This guide covers **EVERYTHING** from scratch. You deleted the old project, so we will rebuild the Android part now.

---

## 🛑 Prerequisites

1.  **Samsung Galaxy Watch** (Wear OS 3.0+) paired with your Android Phone.
2.  **Android Studio** installed.
3.  **Developer Options & WiFi Debugging** enabled on the Watch.
4.  Both Phone and Laptop on the **SAME WiFi Network**.

---

## 🏗️ Phase 1: Set Up the Android Project

1.  Open **Android Studio**.
2.  **New Project**:
    *   Select **Wear OS** on the left.
    *   Choose **Empty Wear App** (or "Empty Activity" for Wear).
    *   Click **Next**.
3.  **Configure**:
    *   **Name**: `AntigravityWear`
    *   **Package Name**: `com.antigravity.wear`
    *   **Language**: `Kotlin`
    *   **Min SDK**: API 30 (Android 11).
    *   Click **Finish** and wait for Gradle Sync.
4.  **Add Mobile Module**:
    *   Go to **File > New > New Module**.
    *   Select **Phone & Tablet** -> **Empty Activity**.
    *   **Module Name**: `mobile`
    *   **Package Name**: `com.antigravity.mobile`
    *   Click **Finish**.

✅ *Result: You now have a project with two modules: `app` (Wear) and `mobile` (Phone).*

---

## ⌚ Phase 2: The Wear OS App (Sender)

This app reads sensors and sends data to the phone via Bluetooth (Data Layer).

### Step 2.1: Dependencies
Open `app/build.gradle.kts` (Module: app) and add inside `dependencies { ... }`:
```kotlin
implementation("com.google.android.gms:play-services-wearable:18.1.0")
```
*Click **Sync Now**.*

### Step 2.2: Permissions
Open `app/src/main/AndroidManifest.xml`. Add these **ABOVE** the `<application>` tag:
```xml
<uses-permission android:name="android.permission.BODY_SENSORS" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

### Step 2.3: The Code
Open `app/src/main/java/com/antigravity/wear/MainActivity.kt`.
**Replace EVERYTHING with this code:**

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
        // Simple checking, normally you'd use a layout
        
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)
        
        // NOTE: In a real app, you must request Runtime Permissions for BODY_SENSORS here.
        // For testing, go to Watch Settings > Apps > AntigravityWear > Permissions > Sensors > Allow.
    }

    override fun onResume() {
        super.onResume()
        heartRateSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_HEART_RATE) {
            val heartRate = event.values[0]
            Log.d("WearApp", "Sending HR: $heartRate")
            sendDataToPhone(heartRate)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun sendDataToPhone(hr: Float) {
        // Send data to /heart_rate path
        val putDataMapReq = PutDataMapRequest.create("/heart_rate")
        putDataMapReq.dataMap.putFloat("bpm", hr)
        putDataMapReq.dataMap.putLong("timestamp", System.currentTimeMillis())
        val putDataReq = putDataMapReq.asPutDataRequest().setUrgent()
        Wearable.getDataClient(this).putDataItem(putDataReq)
    }
}
```

---

## 📱 Phase 3: The Mobile App (Bridge)

This app receives the Bluetooth data and forwards it to your PC via WiFi.

### Step 3.1: Dependencies
Open `mobile/build.gradle.kts` (Module: mobile). Add these:
```kotlin
implementation("com.google.android.gms:play-services-wearable:18.1.0")
implementation("com.squareup.okhttp3:okhttp:4.10.0")
```
*Click **Sync Now**.*

### Step 3.2: The Code
Open `mobile/src/main/java/com/antigravity/mobile/MainActivity.kt`.
**Replace EVERYTHING with this code** (Wait for Step 4 to fill in the IP!):

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
    // ⚠️ TODO: We will update this IP in the next step
    private val SERVER_URL = "ws://192.168.0.16:8080" 

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main) // Ensure 'activity_main.xml' exists (default)
        connectWebSocket()
    }

    override fun onResume() {
        super.onResume()
        Wearable.getDataClient(this).addListener(this)
    }

    override fun onPause() {
        super.onPause()
        Wearable.getDataClient(this).removeListener(this)
    }

    override fun onDataChanged(dataEvents: DataEventBuffer) {
        for (event in dataEvents) {
            if (event.type == DataEvent.TYPE_CHANGED) {
                val path = event.dataItem.uri.path
                if (path == "/heart_rate") {
                    val dataMap = DataMapItem.fromDataItem(event.dataItem).dataMap
                    val hr = dataMap.getFloat("bpm")
                    Log.d("MobileBridge", "Got HR: $hr")
                    sendToWeb(hr)
                }
            }
        }
    }

    private fun connectWebSocket() {
        val client = OkHttpClient()
        val request = Request.Builder().url(SERVER_URL).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                Log.d("MobileBridge", "Connected to PC!")
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                Log.e("MobileBridge", "Connection Failed: ${t.message}")
            }
        })
    }

    private fun sendToWeb(hr: Float) {
        // Send JSON format expected by Web App
        val json = """{"heartRate": $hr}"""
        webSocket?.send(json)
    }
}
```

---

## 🌉 Phase 4: The Bridge Server (PC)

The browser cannot talk to the phone directly easily, so we use a node script as a middleman.

1.  **Create Script**: Create `scripts/socket_bridge.js` in your web project:
    ```javascript
    import { WebSocketServer } from 'ws';
    import os from 'os';

    const wss = new WebSocketServer({ port: 8080 });
    
    // Find IP
    const interfaces = os.networkInterfaces();
    let ip = '0.0.0.0';
    for (const k in interfaces) {
        for (const i of interfaces[k]) {
            if (i.family === 'IPv4' && !i.internal) ip = i.address;
        }
    }

    console.log(`📡 Bridge running on ws://${ip}:8080`);

    wss.on('connection', (ws) => {
        ws.on('message', (msg) => {
            // Broadcast to Web Dashboard
            wss.clients.forEach(c => {
                if (c !== ws && c.readyState === 1) c.send(msg.toString());
            });
        });
    });
    ```
2.  **Run It**:
    ```bash
    node scripts/socket_bridge.js
    ```
3.  **Get IP**: Look at the terminal. It will say something like `ws://192.168.0.16:8080`.
4.  **Update Android Code**: Go back to `MainActivity.kt` in the Mobile App and update `SERVER_URL` with this EXACT address.

---

## 🚀 Phase 5: Launch Everything

1.  **Web App**:
    *   `npm run dev`
    *   Go to **Settings** > **Device Connection** > Toggle **ON** "Samsung Watch Integration".
2.  **Bridge Server**:
    *   Ensure `node scripts/socket_bridge.js` is running.
3.  **Mobile App**:
    *   Connect Phone via USB.
    *   Run `mobile` module.
    *   *Check Logcat: Should say "Connected to PC!".*
4.  **Wear App**:
    *   Connect Watch.
    *   Run `app` module.
    *   *Check Logcat: Should say "Sending HR: ...".*

🎉 **Success!** You should now see the Heart Rate graph moving on your PC.
