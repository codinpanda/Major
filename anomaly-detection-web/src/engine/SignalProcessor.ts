import type { HealthDataPacket } from '../simulation/DataGenerator';

// SOS Coefficients for 5th Order Butterworth (0.5-12Hz, Fs=700)
// Generated via Scipy
const SOS_COEFFS = [
    [0.0000003112448456, 0.0000006224896911, 0.0000003112448456, 1.0000000000000000, -1.8427528251344114, 0.8523958295962411],
    [1.0000000000000000, 2.0000000000000000, 1.0000000000000000, 1.0000000000000000, -1.9296181977477784, 0.9407120744664865],
    [1.0000000000000000, 0.0000000000000000, -1.0000000000000000, 1.0000000000000000, -1.9012993242962646, 0.9017593783290911],
    [1.0000000000000000, -2.0000000000000000, 1.0000000000000000, 1.0000000000000000, -1.9925983679409827, 0.9926207151542771],
    [1.0000000000000000, -2.0000000000000000, 1.0000000000000000, 1.0000000000000000, -1.9973970162384596, 0.9974174107321006],
];

const FS = 700;
const WINDOW_SIZE = 3500; // 5 seconds
const FRAME_LEN = 350;    // 0.5s
const FRAME_STEP = 175;   // 50% overlap

class SOSFilter {
    process(input: number[]): number[] {

        // Clone state for this batch to simulate continuous streaming if needed
        // For now, we filter the whole window "offline" manner like the notebook
        // BUT for real real-time, we should maintain state. 
        // Given we re-process the whole 3500 window every step in this simple implementation:
        // We will reset state for each window to match "Training logic" which filters window-by-window?
        // Actually training filters the WHOLE signal then segments.
        // Approximating: Filter the window with zero input state (transient error at start of window).
        // Better: Maintain state across windows.

        // Let's effectively filter in-place but keep state
        let y = [...input];

        for (let s = 0; s < 5; s++) {
            const b = [SOS_COEFFS[s][0], SOS_COEFFS[s][1], SOS_COEFFS[s][2]];
            const a = [SOS_COEFFS[s][3], SOS_COEFFS[s][4], SOS_COEFFS[s][5]]; // a0 is 1.0

            let w1 = 0; // Delay element 1
            let w2 = 0; // Delay element 2
            // NOTE: Python uses 'sosfilt'. We implement Direct Form II
            // w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
            // y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]

            for (let i = 0; i < y.length; i++) {
                const x_n = y[i];
                const w_n = x_n - a[1] * w1 - a[2] * w2;
                y[i] = b[0] * w_n + b[1] * w1 + b[2] * w2;
                w2 = w1;
                w1 = w_n;
            }
        }
        return y;
    }
}

export class SignalProcessor {
    // 7 Channels: ECG, EDA, RESP, ACCx, ACCy, ACCz, BVP
    private buffers: number[][] = [[], [], [], [], [], [], []];

    // Check if we have enough data
    isReady(): boolean {
        return this.buffers[0].length >= WINDOW_SIZE;
    }

    push(packet: HealthDataPacket) {
        // Ensure packet has arrays (DataGenerator now provides them)
        const len = packet.rawECG?.length || 0;
        if (len === 0) return;

        // Push data to buffers
        // packet.raw... are arrays
        this.buffers[0].push(...(packet.rawECG || []));
        this.buffers[1].push(...(packet.rawEDA || []));
        this.buffers[2].push(...(packet.rawResp || []));
        this.buffers[3].push(...(packet.rawACC_x || []));
        this.buffers[4].push(...(packet.rawACC_y || []));
        this.buffers[5].push(...(packet.rawACC_z || []));
        this.buffers[6].push(...(packet.rawBVP || []));

        // Maintain Rolling Window (Drop old data)
        // We keep slightly more than window to prevent jitter
        const MAX_BUF = WINDOW_SIZE + 700;
        if (this.buffers[0].length > MAX_BUF) {
            for (let i = 0; i < 7; i++) {
                this.buffers[i] = this.buffers[i].slice(this.buffers[i].length - WINDOW_SIZE); // Keep exactly 3500
            }
        }
    }

    getSequence(): Float32Array {
        if (!this.isReady()) throw new Error("Buffer not ready");

        // Grab last 3500 samples
        const channels: number[][] = [];
        for (let i = 0; i < 7; i++) {
            channels.push(this.buffers[i].slice(this.buffers[i].length - WINDOW_SIZE));
        }

        // 1. Filter (SOS)
        const filter = new SOSFilter(); // Create fresh to simulated per-window processing like batch
        for (let i = 0; i < 7; i++) {
            channels[i] = filter.process(channels[i]);
        }

        // 2. Normalize (Z-Score)
        for (let i = 0; i < 7; i++) {
            channels[i] = this.zScore(channels[i]);
        }

        // 3. Sub-windowing & Feature Extraction
        // Output: (19, 28) flattened -> 532 floats
        const features: number[] = [];

        // 3500 data, window 350, step 175
        // i goes 0 to 3500-350, step 175
        // expected count ~19
        for (let j = 0; j <= WINDOW_SIZE - FRAME_LEN; j += FRAME_STEP) {
            // Processing Frame j
            for (let ch = 0; ch < 7; ch++) {
                const frame = channels[ch].slice(j, j + FRAME_LEN);
                features.push(...this.extractFeatures(frame));
            }
        }

        return new Float32Array(features);
    }

    private zScore(data: number[]): number[] {
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length;
        const std = Math.sqrt(variance) + 1e-6;
        return data.map(x => (x - mean) / std);
    }

    private extractFeatures(frame: number[]): number[] {
        // 1. RMSSD
        let sumDiffSq = 0;
        for (let i = 1; i < frame.length; i++) {
            sumDiffSq += Math.pow(frame[i] - frame[i - 1], 2);
        }
        const rmssd = Math.sqrt(sumDiffSq / (frame.length - 1));

        // 2. Energy
        const energy = frame.reduce((a, b) => a + b * b, 0);

        // Freq Domain Preparation (PSD)
        const psd = this.computePSD(frame);

        // 3. Entropy (Spectral)
        const psdSum = psd.reduce((a, b) => a + b, 0) + 1e-10;
        const normPSD = psd.map(p => p / psdSum);
        const entropy = -normPSD.reduce((a, p) => a + (p * Math.log(p + 1e-10)), 0);

        // 4. Dom Freq
        let maxP = -1;
        let maxIdx = 0;
        for (let i = 0; i < psd.length; i++) {
            if (psd[i] > maxP) {
                maxP = psd[i];
                maxIdx = i;
            }
        }
        const domFreq = (maxIdx * FS) / frame.length; // maxIdx is bin

        return [rmssd, energy, entropy, domFreq];
    }

    private computePSD(frame: number[]): number[] {
        // Simple Real DFT (O(N^2) but N=350 is small enough ~122k ops)
        // Optimized: Only compute magnitude squared
        const N = frame.length;
        const psd = new Array(Math.floor(N / 2) + 1).fill(0);

        for (let k = 0; k < psd.length; k++) {
            let re = 0;
            let im = 0;
            const theta_k = (2 * Math.PI * k) / N;

            for (let n = 0; n < N; n++) {
                const angle = theta_k * n;
                re += frame[n] * Math.cos(angle);
                im -= frame[n] * Math.sin(angle);
            }
            // Welch would average segments, here just periodogram of frame
            psd[k] = (re * re + im * im) / N; // |X[k]|^2 / N
        }
        return psd;
    }
}
