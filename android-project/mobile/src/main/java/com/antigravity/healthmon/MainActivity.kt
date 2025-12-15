package com.antigravity.healthmon

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.wearable.DataClient
import com.google.android.gms.wearable.DataEvent
import com.google.android.gms.wearable.DataEventBuffer
import com.google.android.gms.wearable.Wearable
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener

class MainActivity : AppCompatActivity(), DataClient.OnDataChangedListener {

    private lateinit var webSocket: WebSocket
    private val client = OkHttpClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Connect to Web Dashboard (Change IP to your laptop's IP)
        val request = Request.Builder().url("ws://192.168.1.100:8080").build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                Log.d("Antigravity", "Connected to Web Dashboard")
            }
        })
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
            if (event.type == DataEvent.TYPE_CHANGED && event.dataItem.uri.path == "/health_data") {
                val item = com.google.android.gms.wearable.DataMapItem.fromDataItem(event.dataItem)
                val json = item.dataMap.getString("json")
                
                // Forward to Web
                json?.let { webSocket.send(it) }
            }
        }
    }
}
