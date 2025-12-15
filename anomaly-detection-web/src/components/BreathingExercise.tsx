import { useState, useEffect } from 'react';
import { X, Play, Pause } from 'lucide-react';

type BreathingMode = '4-7-8' | 'box';

interface BreathingExerciseProps {
    isOpen: boolean;
    onClose: () => void;
    mode?: BreathingMode;
}

export function BreathingExercise({ isOpen, onClose, mode = '4-7-8' }: BreathingExerciseProps) {
    const [isActive, setIsActive] = useState(false);
    const [phase, setPhase] = useState<'inhale' | 'hold' | 'exhale' | 'rest'>('inhale');
    const [countdown, setCountdown] = useState(4);
    const [cycle, setCycle] = useState(0);

    const modes = {
        '4-7-8': {
            name: '4-7-8 Breathing',
            description: 'Inhale for 4, hold for 7, exhale for 8',
            phases: {
                inhale: 4,
                hold: 7,
                exhale: 8,
                rest: 0,
            },
        },
        'box': {
            name: 'Box Breathing',
            description: 'Inhale, hold, exhale, hold - each for 4 seconds',
            phases: {
                inhale: 4,
                hold: 4,
                exhale: 4,
                rest: 4,
            },
        },
    };

    const currentMode = modes[mode];

    useEffect(() => {
        if (!isActive) return;

        const timer = setInterval(() => {
            setCountdown((prev) => {
                if (prev <= 1) {
                    // Move to next phase
                    if (phase === 'inhale') {
                        setPhase('hold');
                        return currentMode.phases.hold;
                    } else if (phase === 'hold') {
                        setPhase('exhale');
                        return currentMode.phases.exhale;
                    } else if (phase === 'exhale') {
                        if (mode === 'box') {
                            setPhase('rest');
                            return currentMode.phases.rest;
                        } else {
                            setPhase('inhale');
                            setCycle((c) => c + 1);
                            return currentMode.phases.inhale;
                        }
                    } else {
                        // rest phase (box breathing only)
                        setPhase('inhale');
                        setCycle((c) => c + 1);
                        return currentMode.phases.inhale;
                    }
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(timer);
    }, [isActive, phase, mode, currentMode]);

    const handleStart = () => {
        setIsActive(true);
        setPhase('inhale');
        setCountdown(currentMode.phases.inhale);
        setCycle(0);
    };

    const handlePause = () => {
        setIsActive(false);
    };

    const handleReset = () => {
        setIsActive(false);
        setPhase('inhale');
        setCountdown(currentMode.phases.inhale);
        setCycle(0);
    };

    if (!isOpen) return null;

    const getCircleScale = () => {
        if (phase === 'inhale') return 1.5;
        if (phase === 'exhale') return 0.8;
        return 1.2;
    };

    const getPhaseColor = () => {
        if (phase === 'inhale') return '#00C7BE';
        if (phase === 'hold' || phase === 'rest') return '#FFD60A';
        return '#30D158';
    };

    const getPhaseText = () => {
        if (phase === 'inhale') return 'Breathe In';
        if (phase === 'hold') return 'Hold';
        if (phase === 'exhale') return 'Breathe Out';
        return 'Rest';
    };

    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
            <div className="card-base bg-surface max-w-md w-full p-8 relative">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 size-8 rounded-full bg-background hover:bg-surfaceHover flex items-center justify-center text-secondary transition-colors"
                >
                    <X size={20} />
                </button>

                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-2xl font-bold text-primary">{currentMode.name}</h2>
                    <p className="text-sm text-secondary mt-2">{currentMode.description}</p>
                </div>

                {/* Breathing Circle */}
                <div className="flex items-center justify-center mb-8 h-64">
                    <div className="relative">
                        <div
                            className="size-48 rounded-full transition-all duration-[4000ms] ease-in-out flex items-center justify-center"
                            style={{
                                transform: `scale(${getCircleScale()})`,
                                backgroundColor: `${getPhaseColor()}20`,
                                border: `4px solid ${getPhaseColor()}`,
                            }}
                        >
                            <div className="text-center">
                                <div className="text-6xl font-bold text-primary">{countdown}</div>
                                <div className="text-sm text-secondary mt-2">{getPhaseText()}</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Cycle Counter */}
                <div className="text-center mb-6">
                    <div className="text-sm text-secondary">Cycle {cycle + 1}</div>
                </div>

                {/* Controls */}
                <div className="flex gap-3">
                    {!isActive ? (
                        <button
                            onClick={handleStart}
                            className="flex-1 px-6 py-3 bg-accent text-surface rounded-xl font-medium hover:bg-accentDark transition-colors flex items-center justify-center gap-2"
                        >
                            <Play size={20} />
                            Start
                        </button>
                    ) : (
                        <button
                            onClick={handlePause}
                            className="flex-1 px-6 py-3 bg-warning text-surface rounded-xl font-medium hover:bg-warning/90 transition-colors flex items-center justify-center gap-2"
                        >
                            <Pause size={20} />
                            Pause
                        </button>
                    )}
                    <button
                        onClick={handleReset}
                        className="px-6 py-3 bg-background text-primary rounded-xl font-medium hover:bg-surfaceHover transition-colors"
                    >
                        Reset
                    </button>
                </div>

                {/* Tips */}
                <div className="mt-6 p-4 bg-accent/5 rounded-xl border border-accent/10">
                    <div className="text-xs text-secondary">
                        <strong className="text-primary">Tip:</strong> Find a comfortable position and focus on your breath.
                        Complete 3-5 cycles for best results.
                    </div>
                </div>
            </div>
        </div>
    );
}
