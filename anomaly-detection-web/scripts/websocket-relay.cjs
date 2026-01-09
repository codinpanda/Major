const WebSocket = require('ws');

// Standard Node.js WebSocket Server
// Acts as the bridge between the Phone App and the Browser Dashboard
const PORT = 8080;
const wss = new WebSocket.Server({ port: PORT });

console.log(`📡 WebSocket Relay Server running on ws://localhost:${PORT}`);
console.log('   (Waiting for Phone App to connect...)');

wss.on('connection', (ws) => {
    console.log('✅ New Client Connected');

    ws.on('message', (message) => {
        // Broadcast incoming message (from Phone) to all other clients (Browser)
        try {
            // Check if it looks like JSON before broadcasting
            const strMsg = message.toString();
            JSON.parse(strMsg);

            wss.clients.forEach((client) => {
                if (client !== ws && client.readyState === WebSocket.OPEN) {
                    client.send(strMsg);
                }
            });
        } catch (e) {
            console.error('❌ Data error:', e.message);
        }
    });

    ws.on('close', () => {
        console.log('❌ Client Disconnected');
    });
});
