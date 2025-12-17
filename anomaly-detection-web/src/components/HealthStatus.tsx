import clsx from 'clsx';
import { motion } from 'framer-motion';
import type { HealthDataPacket } from '../simulation/DataGenerator';

interface HealthStatusProps {
    latest: HealthDataPacket | null;
    anomalyScore?: number; // Added: derived from ML model
}

export function HealthStatus({ latest, anomalyScore = 0 }: HealthStatusProps) {
    if (!latest) return null;

    // Flagship: Composite Health Score Calculation (0-100)
    // Baseline: 100. Deduct based on Stress and HR deviation.
    // This gives a single "number" for the user to focus on.

    // 1. Stress Penalty: -0.5 points per stress unit > 20
    const stressPenalty = Math.max(0, (latest.stress - 20) * 0.5);

    // 2. HR Penalty: -1 point per beat > 100 (resting assumption)
    const hrPenalty = Math.max(0, (latest.heartRate - 100) * 0.5);

    // 3. Anomaly Penalty: -20 if anomaly suspected
    const anomalyPenalty = (anomalyScore > 0.5) ? 20 * anomalyScore : 0;

    const rawScore = 100 - stressPenalty - hrPenalty - anomalyPenalty;
    const healthScore = Math.max(0, Math.min(100, Math.round(rawScore)));

    // Variables for render
    let status: 'GREAT' | 'GOOD' | 'ATTENTION' = 'GREAT';
    let message = "";
    let subMessage = "";
    let colorClass = "";
    let iconColor = "";

    // Determine State based on Score
    if (healthScore >= 80) {
        status = 'GREAT';
        message = "Peak Condition";
        subMessage = "Your vitals are optimal.";
        colorClass = "from-emerald-500/20 to-emerald-500/5 border-emerald-500/20";
        iconColor = "text-emerald-500";
    } else if (healthScore >= 60) {
        status = 'GOOD';
        message = "Good Condition";
        subMessage = "Slight elevation detected.";
        colorClass = "from-amber-500/20 to-amber-500/5 border-amber-500/20";
        iconColor = "text-amber-500";
    } else {
        status = 'ATTENTION';
        message = "Needs Attention";
        subMessage = "Vitals are outside normal range.";
        colorClass = "from-red-500/20 to-red-500/5 border-red-500/20";
        iconColor = "text-red-500";
    }

    // Force specific overrides for critical anomalies
    if (anomalyScore > 0.8) {
        status = 'ATTENTION';
        message = "Anomaly Detected";
        subMessage = `AI Confidence: ${(anomalyScore * 100).toFixed(0)}%`;
        colorClass = "from-red-500/20 to-red-500/5 border-red-500/20";
        iconColor = "text-red-500";
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={clsx(
                "relative overflow-hidden rounded-3xl p-6 border transition-all duration-500",
                "bg-gradient-to-br",
                colorClass
            )}
        >
            {/* Background Glow */}
            <div className={`absolute -right-10 -top-10 size-40 rounded-full blur-3xl opacity-20 ${status === 'ATTENTION' ? 'bg-red-500' : status === 'GOOD' ? 'bg-amber-500' : 'bg-emerald-500'
                }`} />

            <div className="relative flex items-center gap-6">
                {/* Score Visualizer */}
                <div className="relative size-20 sm:size-24 flex-shrink-0">
                    <svg className="size-full -rotate-90" viewBox="0 0 36 36">
                        {/* Background Ring */}
                        <path className="text-surface" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3" strokeOpacity="0.1" />
                        {/* Progress Ring */}
                        <path
                            className={clsx("transition-all duration-1000 ease-out", iconColor)}
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                            strokeDasharray={`${healthScore}, 100`}
                        />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-2xl sm:text-3xl font-bold text-primary">{healthScore}</span>
                        <span className="text-[10px] uppercase font-bold text-secondary tracking-wider">Score</span>
                    </div>
                </div>

                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <div className={clsx("size-2 rounded-full", iconColor.replace('text-', 'bg-'))}></div>
                        <span className={clsx("text-xs font-bold uppercase tracking-wider", iconColor)}>
                            {status === 'GREAT' ? 'Optimal' : status}
                        </span>
                    </div>
                    <h2 className="text-xl sm:text-2xl font-bold text-primary tracking-tight truncate">{message}</h2>
                    <p className="text-sm text-secondary/90 mt-1 font-medium leading-snug">{subMessage}</p>
                </div>
            </div>
        </motion.div>
    );
}
