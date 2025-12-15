import * as ort from 'onnxruntime-web';
import type { HealthDataPacket } from '../simulation/DataGenerator';

export class InferenceEngine {
    private session: ort.InferenceSession | null = null;
    public isLoaded = false;

    private dataBuffer: number[][] = [];
    private readonly SEQUENCE_LENGTH = 60;

    async init() {
        try {
            // Try to load model, if fail, we use fallback
            this.session = await ort.InferenceSession.create('/model.onnx', {
                executionProviders: ['wasm'],
            });
            this.isLoaded = true;
            console.log("ONNX Model (Hybrid LSTM-GRU) Loaded Successfully");
        } catch (e) {
            console.warn("ONNX Model failed to load (using Rule-Based Fallback)", e);
            this.isLoaded = false;
        }
    }

    async predict(packet: HealthDataPacket): Promise<number> {
        // 1. Buffer Data
        // Features: [HeartRate, HRV, SpO2, Motion]
        // Note: Packet might not have SpO2 explicitly, we derive or use stress as proxy if needed, 
        // but let's assume we update DataGenerator to include it or we map stress -> SpO2 inverse proxy for demo.

        // For accurate mapping to the trained model:
        // Model expects: [HR, HRV, SpO2, Motion]

        // Simple proxy for SpO2 if missing (Stress up -> SpO2 down slightly)
        const proxySpO2 = 98 - (packet.stress / 20);

        const accMag = Math.sqrt(
            packet.accelerometer.x ** 2 +
            packet.accelerometer.y ** 2 +
            packet.accelerometer.z ** 2
        );

        const featureVector = [packet.heartRate, packet.hrv, proxySpO2, accMag];

        this.dataBuffer.push(featureVector);
        if (this.dataBuffer.length > this.SEQUENCE_LENGTH) {
            this.dataBuffer.shift(); // Remove oldest
        }

        // Only predict if we have enough data
        if (this.dataBuffer.length < this.SEQUENCE_LENGTH) {
            return 0; // Warming up
        }

        // FALLBACK: Rule-Based Logic
        if (!this.session || !this.isLoaded) {
            // Anomaly if HR > 110 OR Stress > 80 OR (HRV < 20 & HR > 90)
            const isAbnormal =
                packet.heartRate > 110 ||
                packet.stress > 80 ||
                (packet.hrv < 20 && packet.heartRate > 90);

            return isAbnormal ? 0.95 : 0.05;
        }

        try {
            // Prepare Input Tensor: [1, 60, 4]
            const flatData = this.dataBuffer.flat();
            const inputData = Float32Array.from(flatData);

            const tensor = new ort.Tensor('float32', inputData, [1, this.SEQUENCE_LENGTH, 4]);

            // Run inference
            const feeds: Record<string, ort.Tensor> = {};
            feeds[this.session.inputNames[0]] = tensor;

            const results = await this.session.run(feeds);

            // Output is LogSoftmax: [1, 2] -> [[log_prob_normal, log_prob_anomaly]]
            // We want probability of anomaly
            const outputData = results[this.session.outputNames[0]].data as Float32Array;
            const logProbAnomaly = outputData[1];
            const probAnomaly = Math.exp(logProbAnomaly);

            return probAnomaly;

        } catch (e) {
            console.error("Inference Error (Fallback Activated)", e);
            const isAbnormal = packet.heartRate > 110 || packet.stress > 80;
            return isAbnormal ? 0.95 : 0.05;
        }
    }
}

export const inferenceEngine = new InferenceEngine();
