import { useEffect, useState, useRef, useCallback } from 'react';
import { AlertTriangle, Phone, X, Wind } from 'lucide-react';
import type { HealthDataPacket } from '../simulation/DataGenerator';

interface GuidanceOverlayProps {
    latestData: HealthDataPacket | null;
    anomalyScore?: number;
}

export function GuidanceOverlay({ latestData, anomalyScore = 0 }: GuidanceOverlayProps) {
    const [activeAlert, setActiveAlert] = useState<'PANIC' | 'HIGH_HR' | 'ML_ANOMALY' | null>(null);
    const [showBreathing, setShowBreathing] = useState(false);
    const [isDismissed, setIsDismissed] = useState(false);

    // Audio Context Ref to prevent max-context limit errors
    const audioCtxRef = useRef<AudioContext | null>(null);

    // Audio Alert Function
    const playAlertSound = useCallback(() => {
        try {
            const CtxClass = window.AudioContext || (window as any).webkitAudioContext;
            if (!CtxClass) return;

            // Reuse context or create if missing/closed
            if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
                audioCtxRef.current = new CtxClass();
            }

            const ctx = audioCtxRef.current;
            // Resume if suspended (browser autopilot policy)
            if (ctx.state === 'suspended') {
                ctx.resume();
            }

            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
            osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.5); // Drop pitch

            gain.gain.setValueAtTime(0.1, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);

            osc.start();
            osc.stop(ctx.currentTime + 0.5);
        } catch (e) {
            console.error("Audio play failed", e);
        }
    }, []);

    useEffect(() => {
        if (!latestData) return;

        let newAlert: 'PANIC' | 'HIGH_HR' | 'ML_ANOMALY' | null = null;

        // Prioritize ML Anomaly
        if (anomalyScore > 0.8) {
            newAlert = 'ML_ANOMALY';
        } else if (latestData.heartRate > 115 && latestData.stress > 75) {
            newAlert = 'PANIC';
        } else if (latestData.heartRate > 120) {
            newAlert = 'HIGH_HR';
        }

        // Logic: 
        // 1. If alert changes type (e.g. HIGH_HR -> PANIC), un-dismiss and play sound.
        // 2. If alert clears (null), un-dismiss so next one shows.
        // 3. If alert stays same, respect existing dismissed state.

        if (newAlert !== activeAlert) {
            if (newAlert) {
                // New or escalated alert
                setIsDismissed(false);
                playAlertSound();
            } else {
                // Alert cleared
                setIsDismissed(false);
            }
            setActiveAlert(newAlert);
        }
    }, [latestData, anomalyScore, activeAlert, playAlertSound]);

    // Render Logic:
    // - Show Breathing Modal if active (top priority)
    // - Show Alert Card if active AND not dismissed AND not breathing

    const shouldShowAlert = activeAlert && !isDismissed && !showBreathing;

    if (!shouldShowAlert && !showBreathing) return null;

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-4 max-w-sm w-full animate-slide-up">
            {/* Critical Alert Card */}
            {shouldShowAlert && (
                <div className="bg-surface border border-critical/50 shadow-lg shadow-critical/10 rounded-xl overflow-hidden relative group">
                    {/* Dismiss Button */}
                    <button
                        onClick={() => setIsDismissed(true)}
                        className="absolute top-2 right-2 p-1 text-secondary hover:text-primary hover:bg-white/10 rounded-full transition-colors z-10"
                        title="Dismiss alert"
                    >
                        <X size={16} />
                    </button>

                    <div className="bg-critical/10 p-4 border-b border-critical/20 flex items-start gap-3">
                        <AlertTriangle className="text-critical shrink-0" size={24} />
                        <div className="pr-6"> {/* Padding for dismiss button */}
                            <h3 className="text-danger font-bold text-lg leading-tight">
                                {activeAlert === 'ML_ANOMALY' ? "AI Anomaly Detected" : "Unusual Activity Detected"}
                            </h3>
                            <p className="text-secondary text-sm mt-1">
                                {activeAlert === 'ML_ANOMALY'
                                    ? `Our AI model is ${Math.round(anomalyScore * 100)}% confident this is an anomaly.`
                                    : activeAlert === 'PANIC'
                                        ? "Your heart rate and stress are unusually high together."
                                        : "Your heart rate is higher than your normal baseline."}
                            </p>
                        </div>
                    </div>

                    <div className="p-4 bg-surface space-y-3">
                        <button
                            onClick={() => setShowBreathing(true)}
                            className="w-full flex items-center justify-center gap-2 bg-accent text-surface py-3 rounded-xl font-bold transition-transform active:scale-95 shadow-lg shadow-accent/20"
                        >
                            <Wind size={20} />
                            Let's take a short break
                        </button>

                        <button
                            className="w-full flex items-center justify-center gap-2 border border-secondary/20 text-secondary hover:text-primary hover:bg-white/5 py-3 rounded-xl font-medium transition-colors"
                            onClick={() => {
                                alert(`Calling trusted contact...\nGPS: 37.7749, -122.4194\nHR: ${latestData?.heartRate.toFixed(0)} BPM`);
                            }}
                        >
                            <Phone size={18} />
                            Call someone you trust
                        </button>
                    </div>
                </div>
            )}

            {/* Breathing Exercise Modal/Overlay */}
            {showBreathing && (
                <div className="fixed inset-0 bg-background/95 z-[60] flex items-center justify-center p-4 backdrop-blur-sm">
                    <button
                        onClick={() => setShowBreathing(false)}
                        className="absolute top-6 right-6 text-gray-400 hover:text-white transition-colors"
                    >
                        <X size={32} />
                    </button>

                    <div className="text-center space-y-8 max-w-md w-full">
                        <div className="relative size-64 mx-auto flex items-center justify-center">
                            {/* Medical Breathing Animation */}
                            <div className="absolute inset-0 bg-blue-500/10 rounded-full animate-ping [animation-duration:4s]"></div>
                            <div className="absolute inset-4 bg-blue-500/10 rounded-full animate-pulse [animation-duration:4s]"></div>
                            <div className="relative z-10 size-48 bg-surface border-4 border-blue-500 rounded-full flex items-center justify-center shadow-[0_0_40px_rgba(59,130,246,0.2)]">
                                <span className="text-2xl font-medium text-blue-400">Inhale</span>
                            </div>
                        </div>

                        <div>
                            <h2 className="text-2xl font-bold text-white mb-2">Box Breathing Protocol</h2>
                            <p className="text-gray-400">Follow the visual guide to regulate your autonomic nervous system.</p>
                        </div>

                        <div className="grid grid-cols-3 gap-4 text-center">
                            <div className="p-3 rounded bg-surface border border-gray-800">
                                <div className="text-xl font-bold text-white">4s</div>
                                <div className="text-xs text-gray-500">Inhale</div>
                            </div>
                            <div className="p-3 rounded bg-surface border border-gray-800">
                                <div className="text-xl font-bold text-white">4s</div>
                                <div className="text-xs text-gray-500">Hold</div>
                            </div>
                            <div className="p-3 rounded bg-surface border border-gray-800">
                                <div className="text-xl font-bold text-white">4s</div>
                                <div className="text-xs text-gray-500">Exhale</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
