import * as ort from 'onnxruntime-web';
import type { HealthDataPacket } from '../simulation/DataGenerator';
import { SignalProcessor } from './SignalProcessor';

export class InferenceEngine {
    private session: ort.InferenceSession | null = null;
    private isLoading = true;
    private signalProcessor = new SignalProcessor();

    constructor() {
        this.init();
    }

    async init() {
        try {
            // Configure WASM paths to look at root (public folder)
            ort.env.wasm.wasmPaths = '/';

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
        // approximate progress based on 3500 samples needed
        // We can't access buffer length directly easily without exposing it, 
        // but 'isReady' tells us if we are there.
        // For UI feedback, we'll assume "Collecting..." until ready.
        return {
            isCalibrating: this.isLoading || !this.signalProcessor.isReady(),
            samplesCollected: this.signalProcessor.isReady() ? 3500 : 0, // Simplified
            samplesNeeded: 3500
        };
    }

    async predict(packet: HealthDataPacket): Promise<{ probability: number; zScore: number; isAnomaly: boolean }> {
        if (this.isLoading || !this.session) {
            return { probability: 0, zScore: 0, isAnomaly: false };
        }

        // 1. Feed Data
        this.signalProcessor.push(packet);

        // 2. Check Readiness
        if (!this.signalProcessor.isReady()) {
            // Buffer filling...
            return { probability: 0, zScore: 0, isAnomaly: false };
        }

        try {
            // 3. Get Features [1, 19, 28] flattened
            const features = this.signalProcessor.getSequence();

            // 4. Create Tensor
            // Shape: (Batch=1, Seq=19, Feat=28)
            const tensor = new ort.Tensor('float32', features, [1, 19, 28]);

            // 5. Run Inference
            const feeds: Record<string, ort.Tensor> = {};
            const inputName = this.session.inputNames[0];
            feeds[inputName] = tensor;

            const results = await this.session.run(feeds);

            // 6. Output
            const outputName = this.session.outputNames[0];
            const outputTensor = results[outputName];
            const probability = outputTensor.data[0] as number;

            // Thresholding
            const isAnomaly = probability > 0.5;

            // Z-score approximation 
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
