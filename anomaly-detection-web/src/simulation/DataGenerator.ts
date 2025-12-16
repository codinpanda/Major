import wesadData from './wesad_sample.json';

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
    public mode: 'NORMAL' | 'ANOMALY' | 'RANDOM' = 'NORMAL';
    private anomalyDuration = 0;

    // WESAD Playback Indices
    private baselineIdx = 0;
    private stressIdx = 0;

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

    setMode(mode: 'NORMAL' | 'ANOMALY' | 'RANDOM') {
        this.mode = mode;
        this.anomalyDuration = 0; // Reset any active random anomaly
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

}

export const dataGenerator = new DataGenerator();
