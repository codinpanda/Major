import type { HealthDataPacket } from '../simulation/DataGenerator';

export type HealthState = 'NORMAL' | 'WARNING' | 'CRITICAL';

export interface GuidanceAction {
    id: string;
    title: string;
    description: string;
    type: 'breathing' | 'advice' | 'emergency';
    durationSeconds?: number;
}

export interface SystemState {
    status: HealthState;
    message: string;
    recommendedAction?: GuidanceAction;
}

class GuidanceEngine {
    analyze(packet: HealthDataPacket): SystemState {
        const { heartRate, stress, hrv } = packet;

        // Critical Rules
        if (heartRate > 140) {
            return {
                status: 'CRITICAL',
                message: 'Heart Rate Critically High',
                recommendedAction: {
                    id: 'emergency_stop',
                    title: 'Stop Activity Immediately',
                    description: 'Your heart rate is dangerously high. Sit down and seek help if you feel unwell.',
                    type: 'emergency'
                }
            };
        }

        // Warning Rules
        if (heartRate > 110) {
            return {
                status: 'WARNING',
                message: 'Elevated Heart Rate',
                recommendedAction: {
                    id: 'box_breathing',
                    title: 'Perform Box Breathing',
                    description: 'Inhale 4s, Hold 4s, Exhale 4s, Hold 4s.',
                    type: 'breathing',
                    durationSeconds: 60
                }
            };
        }

        if (stress > 80) {
            return {
                status: 'WARNING',
                message: 'High Stress Detected',
                recommendedAction: {
                    id: 'deep_breathing',
                    title: 'Deep Relaxation',
                    description: 'Take deep breaths to lower your cortisol levels.',
                    type: 'breathing',
                    durationSeconds: 120
                }
            };
        }

        if (hrv < 20 && stress > 50) {
            return {
                status: 'WARNING',
                message: 'Low HRV & High Stress',
                recommendedAction: {
                    id: 'rest_moment',
                    title: 'Take a Moment',
                    description: 'Your recovery score is low. Try to reduce cognitive load.',
                    type: 'advice'
                }
            }
        }

        return {
            status: 'NORMAL',
            message: 'Vitals Stable'
        };
    }
}

export const guidanceEngine = new GuidanceEngine();
