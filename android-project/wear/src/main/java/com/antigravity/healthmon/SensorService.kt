package com.antigravity.healthmon

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import com.google.android.gms.wearable.PutDataMapRequest
import com.google.android.gms.wearable.Wearable
import org.json.JSONObject
import java.nio.charset.StandardCharsets

class SensorService : Service(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var heartRate: Float = 0f
    private var lastHrvTimestamp: Long = 0
    private var rrIntervals = mutableListOf<Float>()

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        
        val channel = NotificationChannel("health_mon", "Health Monitor", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        
        val notification = Notification.Builder(this, "health_mon")
            .setContentTitle("Antigravity Monitor")
            .setContentText("Tracking vitals...")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .build()
            
        startForeground(1, notification)

        // Register Sensors
        sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        
        if (event.sensor.type == Sensor.TYPE_HEART_RATE) {
            heartRate = event.values[0]
            // Calculate pseudo-HRV from Heart Rate (Simulated for demo as raw RR is hard to get on all devices)
            // In production, use Samsung Privileged Health SDK for real RR intervals
            val rr = 60000 / heartRate
            rrIntervals.add(rr)
            if (rrIntervals.size > 10) rrIntervals.removeAt(0)
            
            sendDataToPhone()
        }
    }

    private fun sendDataToPhone() {
        // Create JSON Payload
        val json = JSONObject().apply {
            put("heartRate", heartRate)
            put("timestamp", System.currentTimeMillis())
            // ... add accel
        }

        val request = PutDataMapRequest.create("/health_data").run {
            dataMap.putString("json", json.toString())
            asPutDataRequest()
        }
        
        Wearable.getDataClient(this).putDataItem(request)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    override fun onBind(intent: Intent?): IBinder? = null
}
