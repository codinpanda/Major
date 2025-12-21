import wesadData from './wesad_sample.json';
import { WebSocketClient } from './WebSocketClient';

export interface HealthDataPacket {
    timestamp: number;
    heartRate: number;
    hrv: number;
    stress: number;
    accelerometer: { x: number; y: number; z: number };
    // Raw signals for ONNX model (7 Channels Required)
    rawECG?: number[];
    rawEDA?: number[];
    rawResp?: number[];
    rawBVP?: number[];
    rawACC_x?: number[];
    rawACC_y?: number[];
    rawACC_z?: number[];
}

type Listener = (data: HealthDataPacket) => void;

class DataGenerator {
    private listeners: Listener[] = [];
    private intervalId: number | null = null;
    public mode: 'NORMAL' | 'ANOMALY' | 'RANDOM' | 'EXTERNAL_DEVICE' = 'NORMAL';
    private anomalyDuration = 0;
    private timeoutId: number | null = null;

    // External Data Source
    private wsClient: WebSocketClient;

    // WESAD Playback Indices
    private baselineIdx = 0;
    private stressIdx = 0;

    // Synth State
    private timeStep = 0;

    constructor() {
        this.wsClient = new WebSocketClient('ws://localhost:8080', (data) => {
            if (this.mode === 'EXTERNAL_DEVICE') {
                this.emitExternal(data);
            }
        });
    }

    start() {
        if (this.intervalId) return;

        this.intervalId = window.setInterval(() => {
            this.emit();
        }, 100); // 10Hz updates for smoother graph, but chunks are larger
    }

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
    }

    setMode(mode: 'NORMAL' | 'ANOMALY' | 'RANDOM' | 'EXTERNAL_DEVICE') {
        this.mode = mode;
        this.anomalyDuration = 0; // Reset any active random anomaly

        if (mode === 'EXTERNAL_DEVICE') {
            this.wsClient.connect();
        } else {
            this.wsClient.disconnect();
        }
    }

    subscribe(listener: Listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    private emit() {
        const now = Date.now();
        let packet: HealthDataPacket;
        let isAnomalyStr = false;
        let rawSequence: any[] = [];

        // Determine Mode
        if (this.mode === 'RANDOM') {
            if (this.anomalyDuration === 0 && Math.random() < 0.1) {
                this.anomalyDuration = 5 + Math.floor(Math.random() * 5);
            }
            if (this.anomalyDuration > 0) {
                isAnomalyStr = true;
                this.anomalyDuration--;
            }
        } else if (this.mode === 'ANOMALY') {
            isAnomalyStr = true;
        }

        // Get Real WESAD Sequence
        // wesadData structure: { baseline: [[ecg, eda]...], stress: [...] }
        // Each interval we emit a chunk of data. 
        // 700Hz original. If we emit every 100ms, we need 70 samples to keep up.
        // Assuming rawSequence is a chunk.

        const CHUNK_SIZE = 70; // 100ms worth of 700Hz data

        if (!isAnomalyStr) {
            // Normal (Baseline)
            const data = wesadData.baseline || [];
            if (data.length > 0) {
                // Slice a chunk
                const start = this.baselineIdx % (data.length - CHUNK_SIZE);
                rawSequence = data.slice(start, start + CHUNK_SIZE) as any[];
                this.baselineIdx = (this.baselineIdx + CHUNK_SIZE) % data.length;
            }

            packet = {
                timestamp: now,
                heartRate: 60 + Math.random() * 5, // Stable HR
                hrv: 50 + Math.random() * 5,
                stress: 20 + Math.random() * 5,
                accelerometer: { x: 0, y: 0, z: 9.8 }
            };
        } else {
            // Critical (Stress)
            const data = wesadData.stress || [];
            if (data.length > 0) {
                const start = this.stressIdx % (data.length - CHUNK_SIZE);
                rawSequence = data.slice(start, start + CHUNK_SIZE) as any[];
                this.stressIdx = (this.stressIdx + CHUNK_SIZE) % data.length;
            }

            packet = {
                timestamp: now,
                heartRate: 110 + Math.random() * 10, // High HR
                hrv: 15 + Math.random() * 5,
                stress: 90 + Math.random() * 5,
                accelerometer: { x: 0.1, y: 0.1, z: 9.8 }
            };
        }

        // Attach Raw Data for Model (7 Channels)
        if (rawSequence.length > 0) {
            const count = rawSequence.length;

            // 1. ECG & EDA (Real)
            packet.rawECG = rawSequence.map((d: any) => d[0]);
            packet.rawEDA = rawSequence.map((d: any) => d[1]);

            // 2. Synthesize Missing Channels
            packet.rawResp = [];
            packet.rawBVP = [];
            packet.rawACC_x = [];
            packet.rawACC_y = [];
            packet.rawACC_z = [];

            for (let i = 0; i < count; i++) {
                const t = (this.timeStep + i) * (1 / 700); // monotonic time

                // Resp: ~0.25Hz sine wave
                packet.rawResp.push(Math.sin(2 * Math.PI * 0.25 * t));

                // BVP: ~1Hz pulse (matches 60bpm base)
                packet.rawBVP.push(Math.sin(2 * Math.PI * 1.0 * t) * 0.5 + Math.random() * 0.01);

                // ACC: Static or slight noise (Chest ACC)
                // WESAD ACC is usually near 9.8 on one axis.
                packet.rawACC_x.push(0.0 + Math.random() * 0.02);
                packet.rawACC_y.push(0.0 + Math.random() * 0.02);
                packet.rawACC_z.push(0.8 + Math.random() * 0.02); // 0.8g ~ 7.8m/s2
            }
            this.timeStep += count;
        }

        this.listeners.forEach(l => l(packet));

        // Throttle ANOMALY mode to give user time to react (2s pause)
        if (this.mode === 'ANOMALY') {
            this.stop(); // Temporarily stop the interval
            this.timeoutId = window.setTimeout(() => {
                this.timeoutId = null;
                // Only resume if mode hasn't changed
                if (this.mode === 'ANOMALY') {
                    this.start();
                }
            }, 2000);
        }
    }

    private emitExternal(data: any) {
        // Map external JSON (from phone bridge) to internal HealthDataPacket
        // Assuming external sends arrays for raw data? Or we buffer them.
        // For now, minimal support.
        const packet: HealthDataPacket = {
            timestamp: data.timestamp || Date.now(),
            heartRate: data.heartRate || 0,
            hrv: data.hrv || 0,
            stress: 0,
            accelerometer: {
                x: data.accX || 0,
                y: data.accY || 0,
                z: data.accZ || 0
            },
            rawECG: data.rawECG,
            rawEDA: data.rawEDA,
            // Mock missing if needed
            rawResp: data.rawResp || Array(10).fill(0),
            rawBVP: data.rawBVP || Array(10).fill(0),
            rawACC_x: data.rawACC_x || Array(10).fill(0),
            rawACC_y: data.rawACC_y || Array(10).fill(0),
            rawACC_z: data.rawACC_z || Array(10).fill(0),
        };

        this.listeners.forEach(l => l(packet));
    }
}

export const dataGenerator = new DataGenerator();
