const WebSocket = require('ws');

// Configuration
const RELAY_URL = 'ws://localhost:8080';
const UPDATE_INTERVAL_MS = 100; // 10Hz streaming

console.log(`🏃 Starting Terra Simulator...`);
console.log(`📡 Connecting to Backend API: ${RELAY_URL}`);

const ws = new WebSocket(RELAY_URL);

// Simulation State
let timeStep = 0;
let heartRate = 70;
let steps = 0;

ws.on('open', () => {
    console.log('✅ Connected to Relay. Begin streaming...');

    setInterval(() => {
        const packet = generateTerraPacket();
        ws.send(JSON.stringify(packet));

        // Log occasional status
        if (timeStep % 50 === 0) { // Every 5 seconds
            console.log(`📤 [Terra] Sent packet #${timeStep}: HR=${packet.heartRate.toFixed(1)} bpm, AccZ=${packet.accZ.toFixed(2)}`);
        }
    }, UPDATE_INTERVAL_MS);
});

ws.on('error', (err) => {
    console.error(`❌ Connection Error: ${err.message}`);
    console.log("   (Make sure 'node scripts/websocket-relay.js' is running!)");
    process.exit(1);
});

ws.on('close', () => {
    console.log('🔌 Disconnected from Relay.');
    process.exit(0);
});

// --- Data Generation Logic ---
function generateTerraPacket() {
    timeStep++;
    const t = timeStep * (UPDATE_INTERVAL_MS / 1000);

    // 1. Heart Rate Simulation (Normal + occasional stress spikes)
    // Base sine wave + noise
    let targetHR = 70 + 10 * Math.sin(t * 0.1);

    // Inject Anomaly (Tachycardia) every 30 seconds
    if (timeStep % 300 > 250) {
        targetHR = 130 + Math.random() * 20; // > 120 bpm alert condition
    }

    // Inject Anomaly (Bradycardia) every 45 seconds offset
    if (timeStep % 450 > 400 && timeStep % 300 <= 250) {
        targetHR = 40 + Math.random() * 5; // < 45 bpm alert condition
    }

    // Smooth transition
    heartRate = heartRate * 0.9 + targetHR * 0.1;

    // 2. Accelerometer (Gravity + Movement)
    // Z-axis dominates (gravity) + walking motion
    const movement = Math.sin(t * 5) * 2; // Walking cadence
    const accX = Math.random() * 0.2;
    const accY = Math.random() * 0.2;
    const accZ = 9.8 + movement + (Math.random() * 0.5);

    // 3. Steps
    if (movement > 1.5) { // Threshold for step
        steps += 1;
    }

    // 4. Raw Signal Generation (7 Channels for ML Model)
    // The frontend expects raw arrays for inference
    const RAW_SAMPLES = 7; // Number of samples per packet (approx matching 70Hz if we send 10 pkts/s? No, model wants 700Hz. We mock it.)

    const rawECG = Array(RAW_SAMPLES).fill(0).map(() => Math.sin(t * 10) + Math.random() * 0.1);
    const rawEDA = Array(RAW_SAMPLES).fill(0).map(() => 1.0 + Math.sin(t * 0.05));
    const rawResp = Array(RAW_SAMPLES).fill(0).map(() => Math.sin(t * 0.3)); // Breathing
    const rawBVP = Array(RAW_SAMPLES).fill(0).map(() => Math.sin(t * 2));
    const rawACC_x = Array(RAW_SAMPLES).fill(0).map(() => accX);
    const rawACC_y = Array(RAW_SAMPLES).fill(0).map(() => accY);
    const rawACC_z = Array(RAW_SAMPLES).fill(0).map(() => accZ);

    return {
        // Terra / External API Standard Fields
        timestamp: Date.now(),
        heartRate: heartRate,
        hrv: 50 + Math.random() * 10,
        steps: steps,
        accX: accX,
        accY: accY,
        accZ: accZ,

        // Raw Data Buffers for ML Inference
        rawECG, rawEDA, rawResp, rawBVP, rawACC_x, rawACC_y, rawACC_z
    };
}
