import * as ort from 'onnxruntime-web';
import type { HealthDataPacket } from '../simulation/DataGenerator';

export class InferenceEngine {
    private session: ort.InferenceSession | null = null;
    private isLoading = true;

    constructor() {
        this.init();
    }

    async init() {
        try {
            // Load the ONNX model from the public directory
            this.session = await ort.InferenceSession.create('./model/hybrid_model.onnx', {
                executionProviders: ['wasm'],
            });
            this.isLoading = false;
            console.log('ONNX Model loaded successfully');
        } catch (e) {
            console.error('Failed to load ONNX model:', e);
        }
    }

    public getStatus() {
        return {
            isCalibrating: this.isLoading, // Re-using this flag for UI compatibility
            samplesCollected: 0,
            samplesNeeded: 0
        };
    }

    async predict(packet: HealthDataPacket): Promise<{ probability: number; zScore: number; isAnomaly: boolean }> {
        if (this.isLoading || !this.session || !packet.rawECG || !packet.rawEDA) {
            return { probability: 0, zScore: 0, isAnomaly: false };
        }

        try {
            // Prepare Input Tensor [1, 60, 2]
            // Interleave ECG and EDA: [ecg0, eda0, ecg1, eda1, ...]
            const inputData = new Float32Array(packet.rawECG.length * 2);
            for (let i = 0; i < packet.rawECG.length; i++) {
                inputData[i * 2] = packet.rawECG[i];
                inputData[i * 2 + 1] = packet.rawEDA[i];
            }

            const tensor = new ort.Tensor('float32', inputData, [1, packet.rawECG.length, 2]);

            // Run Inference
            const feeds = { input: tensor }; // 'input' matches export name
            const results = await this.session.run(feeds);
            const output = results.output.data[0] as number; // 'output' matches export name

            // Output is Sigmoid probability (0-1)
            const probability = output;
            const isAnomaly = probability > 0.5;

            return {
                probability,
                zScore: probability * 3, // Mock Z-score for UI consistency
                isAnomaly
            };
        } catch (e) {
            console.error('Inference failed:', e);
            return { probability: 0, zScore: 0, isAnomaly: false };
        }
    }
}

export const inferenceEngine = new InferenceEngine();
