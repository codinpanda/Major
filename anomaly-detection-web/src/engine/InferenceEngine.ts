import * as ort from 'onnxruntime-web';
import type { HealthDataPacket } from '../simulation/DataGenerator';

export class InferenceEngine {
    private session: ort.InferenceSession | null = null;
    private isLoading = true;
    private sequenceLength = 60; // Model expects 60 time steps
    private inputDim = 2; // ECG, EDA

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
            isCalibrating: this.isLoading,
            samplesCollected: 0,
            samplesNeeded: 0
        };
    }

    async predict(packet: HealthDataPacket): Promise<{ probability: number; zScore: number; isAnomaly: boolean }> {
        if (this.isLoading || !this.session || !packet.rawECG || !packet.rawEDA) {
            return { probability: 0, zScore: 0, isAnomaly: false };
        }

        // Ensure we have enough data
        if (packet.rawECG.length < this.sequenceLength) {
            console.warn(`Not enough data for inference. Need ${this.sequenceLength}, got ${packet.rawECG.length}`);
            return { probability: 0, zScore: 0, isAnomaly: false };
        }

        try {
            // Prepare Input Tensor [1, 60, 2]
            // Model expects input shape: (batch_size, seq_len, input_size) -> (1, 60, 2)
            // Flattened array should be: [ecg_t0, eda_t0, ecg_t1, eda_t1, ..., ecg_t59, eda_t59]

            const inputData = new Float32Array(this.sequenceLength * this.inputDim);

            for (let i = 0; i < this.sequenceLength; i++) {
                // Use the last 60 points if we have more, or match exactly
                const idx = packet.rawECG.length - this.sequenceLength + i;
                inputData[i * 2] = packet.rawECG[idx];     // Feature 0: ECG
                inputData[i * 2 + 1] = packet.rawEDA[idx]; // Feature 1: EDA
            }

            const tensor = new ort.Tensor('float32', inputData, [1, this.sequenceLength, this.inputDim]);

            // Run Inference
            // Input name MUST match the export. Usually 'input' or 'input.1' from PyTorch export.
            // We can inspect session.inputNames to be sure, but standard PyTorch export usually uses 'input'.
            const feeds: Record<string, ort.Tensor> = {};
            const inputName = this.session.inputNames[0];
            feeds[inputName] = tensor;

            const results = await this.session.run(feeds);

            // Output handling
            // Model returns probability (from Sigmoid in ExportModel wrapper)
            const outputName = this.session.outputNames[0];
            const outputTensor = results[outputName];
            const probability = outputTensor.data[0] as number;

            // Thresholding
            const isAnomaly = probability > 0.5;

            // Simple Z-score approximation for UI (prob 0.5 -> 0 sigma, 0.99 -> 3 sigma)
            // This is just for visualization consistency until we have a real statistical baseline
            const zScore = (probability - 0.5) * 6;

            return {
                probability,
                zScore: Math.max(0, zScore),
                isAnomaly
            };
        } catch (e) {
            console.error('Inference failed:', e);
            return { probability: 0, zScore: 0, isAnomaly: false };
        }
    }
}

export const inferenceEngine = new InferenceEngine();
