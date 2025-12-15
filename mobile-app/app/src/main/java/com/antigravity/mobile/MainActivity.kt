package com.antigravity.mobile

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString

class MainActivity : AppCompatActivity() {
    private var webSocket: WebSocket? = null
    
    // Bridge between Watch and Web
    private val TAG = "AntigravityMobile"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        setupWebSocket()
    }

    private fun setupWebSocket() {
        // Connect to Antigravity Web Dashboard (local or hosted)
        // In a real app, this would forward data received from Wear OS Layer
    }
    
    fun onDataReceivedFromWatch(hr: Int, acc: FloatArray) {
        Log.d(TAG, "Forwarding HR: $hr to Web")
        // webSocket?.send(...)
    }
}
