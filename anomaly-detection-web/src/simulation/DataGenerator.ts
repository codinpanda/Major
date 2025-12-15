export interface HealthDataPacket {
    timestamp: number;
    heartRate: number;
    hrv: number;
    stress: number;
    accelerometer: { x: number; y: number; z: number };
}

type Listener = (data: HealthDataPacket) => void;

class DataGenerator {
    private listeners: Listener[] = [];
    private intervalId: number | null = null;
    public mode: 'NORMAL' | 'ANOMALY' | 'RANDOM' = 'NORMAL';
    private anomalyDuration = 0;

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

        if (this.mode === 'RANDOM') {
            // 10% chance to start anomaly if not active
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

        if (!isAnomalyStr) {
            // Normal
            packet = {
                timestamp: now,
                heartRate: 60 + Math.random() * 25,
                hrv: 45 + Math.random() * 25,
                stress: 15 + Math.random() * 15,
                accelerometer: {
                    x: (Math.random() - 0.5) * 0.05,
                    y: (Math.random() - 0.5) * 0.05,
                    z: 9.8
                }
            };
        } else {
            // Critical
            packet = {
                timestamp: now,
                heartRate: 130 + Math.random() * 30,
                hrv: 15 + Math.random() * 10,
                stress: 85 + Math.random() * 15,
                accelerometer: {
                    x: (Math.random() - 0.5) * 1.5,
                    y: (Math.random() - 0.5) * 1.5,
                    z: 9.8
                }
            };
        }

        this.listeners.forEach(l => l(packet));
    }
}

export const dataGenerator = new DataGenerator();
