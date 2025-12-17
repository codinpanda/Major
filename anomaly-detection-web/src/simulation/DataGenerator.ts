import wesadData from './wesad_sample.json';
import { WebSocketClient } from './WebSocketClient';

export interface HealthDataPacket {
    timestamp: number;
    heartRate: number;
    hrv: number;
    stress: number;
    accelerometer: { x: number; y: number; z: number };
    // Raw signals for ONNX model
    rawECG?: number[];
    rawEDA?: number[];
}

type Listener = (data: HealthDataPacket) => void;

class DataGenerator {
    private listeners: Listener[] = [];
    private intervalId: number | null = null;
    public mode: 'NORMAL' | 'ANOMALY' | 'RANDOM' | 'EXTERNAL_DEVICE' = 'NORMAL';
    private anomalyDuration = 0;

    // External Data Source
    private wsClient: WebSocketClient;

    // WESAD Playback Indices
    // WESAD Playback Indices
    private baselineIdx = 0;
    private stressIdx = 0;

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
        }, 1000);
    }

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
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
        if (!isAnomalyStr) {
            // Normal (Baseline)
            const data = wesadData.baseline || [];
            if (data.length > 0) {
                rawSequence = data[this.baselineIdx % data.length] as any[];
                this.baselineIdx++;
            }

            packet = {
                timestamp: now,
                heartRate: 60 + Math.random() * 20, // Simulated Display HR
                hrv: 50 + Math.random() * 15,
                stress: 20 + Math.random() * 10,
                accelerometer: { x: 0, y: 0, z: 9.8 }
            };
        } else {
            // Critical (Stress)
            const data = wesadData.stress || [];
            if (data.length > 0) {
                rawSequence = data[this.stressIdx % data.length] as any[];
                this.stressIdx++;
            }

            packet = {
                timestamp: now,
                heartRate: 110 + Math.random() * 30, // Simulated Display HR
                hrv: 15 + Math.random() * 10,
                stress: 90 + Math.random() * 10,
                accelerometer: { x: 0.1, y: 0.1, z: 9.8 }
            };
        }

        // Attach Raw Data for Model
        // Data format from python: [ [ecg, eda], [ecg, eda], ... ]
        if (rawSequence.length > 0) {
            packet.rawECG = rawSequence.map((d: any) => d[0]); // Column 0 is ECG
            packet.rawEDA = rawSequence.map((d: any) => d[1]); // Column 1 is EDA
        }

        this.listeners.forEach(l => l(packet));
    }

    private emitExternal(data: any) {
        // Map external JSON (from phone bridge) to internal HealthDataPacket
        const packet: HealthDataPacket = {
            timestamp: data.timestamp || Date.now(),
            heartRate: data.heartRate || 0,
            hrv: data.hrv || 0,
            stress: 0, // Calculated by Inference Engine or not available
            accelerometer: {
                x: data.accX || 0,
                y: data.accY || 0,
                z: data.accZ || 0
            },
            // Pass raw signals if available, otherwise InferenceEngine might skip prediction
            rawECG: data.rawECG,
            rawEDA: data.rawEDA
        };

        this.listeners.forEach(l => l(packet));
    }
}

export const dataGenerator = new DataGenerator();
