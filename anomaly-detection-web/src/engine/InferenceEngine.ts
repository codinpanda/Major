import type { HealthDataPacket } from '../simulation/DataGenerator';

export class InferenceEngine {
    private isCalibrating = true;
    private calibrationBuffer: HealthDataPacket[] = [];
    private readonly CALIBRATION_SIZE = 30; // 30 samples (~30 seconds)

    // Baseline statistics
    private stats = {
        hrMean: 0,
        hrStdDev: 0,
        hrvMean: 0,
        hrvStdDev: 0
    };

    public getStatus() {
        return {
            isCalibrating: this.isCalibrating,
            samplesCollected: this.calibrationBuffer.length,
            samplesNeeded: this.CALIBRATION_SIZE
        };
    }

    async predict(packet: HealthDataPacket): Promise<{ probability: number; zScore: number; isAnomaly: boolean }> {
        if (this.isCalibrating) {
            this.calibrationBuffer.push(packet);

            if (this.calibrationBuffer.length >= this.CALIBRATION_SIZE) {
                this.calculateBaseline();
                this.isCalibrating = false;
            }
            return { probability: 0, zScore: 0, isAnomaly: false };
        }

        // Calculate Z-Scores
        // Z = (Value - Mean) / StdDev
        const zHR = Math.abs((packet.heartRate - this.stats.hrMean) / (this.stats.hrStdDev || 1));
        const zHRV = Math.abs((packet.hrv - this.stats.hrvMean) / (this.stats.hrvStdDev || 1));

        // Combined Anomaly Score (Weighted)
        // HR deviation is critical, HRV deviation is warning
        const weightedScore = (zHR * 0.7) + (zHRV * 0.3);

        // Map Z-Score to Probability (Sigmoid-ish mostly for 0-1 range)
        // Z=2 -> ~95% confidence (2 sigma)
        // Z=3 -> ~99.7% confidence
        // Normalizing: Value < 1.5 -> 0%, Value > 3.5 -> 100%
        const probability = Math.min(1, Math.max(0, (weightedScore - 1.5) / 2));

        // Dynamic Update (Running Average) - "Learning" over time
        // Alpha = 0.05 means new data affects baseline by 5%
        this.updateStats(packet);

        return {
            probability,
            zScore: weightedScore,
            isAnomaly: weightedScore > 2.5 // Threshold: 2.5 Sigma
        };
    }

    private calculateBaseline() {
        const hrs = this.calibrationBuffer.map(p => p.heartRate);
        const hrvs = this.calibrationBuffer.map(p => p.hrv);

        this.stats = {
            hrMean: this.mean(hrs),
            hrStdDev: this.stdDev(hrs),
            hrvMean: this.mean(hrvs),
            hrvStdDev: this.stdDev(hrvs)
        };
        console.log("Model Calibrated:", this.stats);
    }

    private updateStats(packet: HealthDataPacket) {
        // Simple Exponential Moving Average for drift adaptation
        const alpha = 0.01; // Slow adaptation
        this.stats.hrMean = (1 - alpha) * this.stats.hrMean + alpha * packet.heartRate;
        this.stats.hrvMean = (1 - alpha) * this.stats.hrvMean + alpha * packet.hrv;
        // StdDev update simplified for performance, usually kept static or re-calibrated
    }

    private mean(data: number[]) {
        return data.reduce((a, b) => a + b, 0) / data.length;
    }

    private stdDev(data: number[]) {
        const m = this.mean(data);
        const variance = data.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / data.length;
        return Math.sqrt(variance);
    }
}

export const inferenceEngine = new InferenceEngine();
