# Android & Wear OS Development Guide 📱⌚

This guide details how to implement the "Samsung Watch" feature using a standard Android + Wear OS architecture.

## Architecture Overview

**Flow:** `[Samsung Watch]` $\xrightarrow{\text{Bluetooth (Data Layer)}}$ `[Android Phone]` $\xrightarrow{\text{WebSocket (WiFi)}}` `[Web Dashboard]`

1.  **Wear OS App**: Reads Heart Rate & Accelerometer sensors. Broadcasts data to the paired phone using the **Google Play Services Data Layer API**.
2.  **Mobile Companion App**: Listens for Data Layer events. Acts as a bridge, forwarding the received data to the Web App via a **WebSocket** connection.
3.  **Web App**: Listens on a WebSocket for incoming JSON packets and visualizes them.

---

## 1. Wear OS App (Sender)

**Goal**: Read sensors and send to Phone.

### Dependencies (`build.gradle.kts`)
```kotlin
implementation("com.google.android.gms:play-services-wearable:18.1.0")
```

### Implementation (`SensorService.kt`)
Modify your existing service to send data:

```kotlin
// ... inside onSensorChanged ...
if (event.sensor.type == Sensor.TYPE_HEART_RATE) {
    val heartRate = event.values[0]
    sendToPhone(heartRate)
}

// ... helper function ...
private fun sendToPhone(heartRate: Float) {
    val putDataMapRequest = PutDataMapRequest.create("/sensor/heart_rate")
    putDataMapRequest.dataMap.putFloat("heartRate", heartRate)
    putDataMapRequest.dataMap.putLong("timestamp", System.currentTimeMillis())
    val request = putDataMapRequest.asPutDataRequest().setUrgent()
    
    Wearable.getDataClient(this).putDataItem(request)
}
```

---

## 2. Mobile App (Bridge)

**Goal**: Receive from Watch, Send to Web.

### Dependencies (`build.gradle.kts`)
```kotlin
implementation("com.google.android.gms:play-services-wearable:18.1.0")
implementation("com.squareup.okhttp3:okhttp:4.10.0") // For WebSocket
```

### Implementation (`MainActivity.kt` or `BridgeService.kt`)

**A. Listen to Watch:**
Implement `DataClient.OnDataChangedListener`.

```kotlin
override fun onDataChanged(dataEvents: DataEventBuffer) {
    for (event in dataEvents) {
        if (event.type == DataEvent.TYPE_CHANGED) {
            val item = event.dataItem
            if (item.uri.path == "/sensor/heart_rate") {
                val dataMap = DataMapItem.fromDataItem(item).dataMap
                val hr = dataMap.getFloat("heartRate")
                val ts = dataMap.getLong("timestamp")
                
                // Forward to Web
                sendToWeb(hr, ts)
            }
        }
    }
}
```

**B. Send to Web (WebSocket):**
Use OkHttp to connect to your computer's IP (e.g., `ws://192.168.1.X:8080`).

```kotlin
private var webSocket: WebSocket? = null

private fun connectToWeb() {
    val client = OkHttpClient()
    val request = Request.Builder().url("ws://192.168.1.100:8080").build()
    webSocket = client.newWebSocket(request, object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.d("Bridge", "Connected to Web Dashboard")
        }
    })
}

private fun sendToWeb(hr: Float, timestamp: Long) {
    val json = """{"heartRate": $hr, "timestamp": $timestamp}"""
    webSocket?.send(json)
}
```

---

## 3. Web App (Receiver)

**Goal**: Display the data.

The web app needs a `WebSocketClient` (which we are implementing next) to listen for these JSON packets.

**Testing without a Watch:**
You can run `node scripts/mock_mobile_bridge.js` to simulate the generic WebSocket stream that the phone would produce.
