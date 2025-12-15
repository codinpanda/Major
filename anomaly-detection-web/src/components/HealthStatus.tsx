import { Activity, AlertCircle, CheckCircle2 } from 'lucide-react';
import clsx from 'clsx';
import { motion } from 'framer-motion';
import type { HealthDataPacket } from '../simulation/DataGenerator';

interface HealthStatusProps {
    latest: HealthDataPacket | null;
    anomalyScore?: number; // Added: derived from ML model
}

export function HealthStatus({ latest, anomalyScore = 0 }: HealthStatusProps) {
    if (!latest) return null;

    let status: 'GREAT' | 'GOOD' | 'ATTENTION' = 'GREAT';
    let message = "You're excellent!";
    let subMessage = "All systems normal";
    let colorClass = "from-success/20 to-success/5 border-success/20";
    let iconColor = "text-success";

    // Logic: Use ML Prediction if high confidence, otherwise fallback to simple rules
    if (anomalyScore > 0.8) {
        status = 'ATTENTION';
        message = "Anomaly Detected";
        subMessage = `AI Model Confidence: ${(anomalyScore * 100).toFixed(0)}%`;
        colorClass = "from-danger/20 to-danger/5 border-danger/20";
        iconColor = "text-danger";
    } else if (latest.heartRate > 110 || latest.stress > 70) {
        // Fallback checks
        status = 'ATTENTION';
        message = "Attention Needed";
        subMessage = "High heart rate or stress detected";
        colorClass = "from-danger/20 to-danger/5 border-danger/20";
        iconColor = "text-danger";
    } else if (latest.stress > 50 || anomalyScore > 0.5) {
        status = 'GOOD';
        message = "Doing Good";
        subMessage = anomalyScore > 0.5 ? "Slight irregularity detected" : "Stress is slightly elevated";
        colorClass = "from-warning/20 to-warning/5 border-warning/20";
        iconColor = "text-warning";
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
            <div className={`absolute -right-10 -top-10 size-40 rounded-full blur-3xl opacity-20 ${status === 'ATTENTION' ? 'bg-danger' : status === 'GOOD' ? 'bg-warning' : 'bg-success'}`} />

            <div className="relative flex items-center gap-5">
                <div className="relative">
                    {/* Ring Animation */}
                    <div className={clsx(
                        "absolute inset-0 rounded-full animate-ping opacity-20",
                        status === 'ATTENTION' ? 'bg-danger' : status === 'GOOD' ? 'bg-warning' : 'bg-success'
                    )} />

                    <div className={clsx(
                        "relative size-16 rounded-full flex items-center justify-center bg-surface border-4 shadow-xl",
                        status === 'ATTENTION' ? 'border-danger/20' : status === 'GOOD' ? 'border-warning/20' : 'border-success/20'
                    )}>
                        {status === 'GREAT' && <CheckCircle2 size={32} className={iconColor} />}
                        {status === 'GOOD' && <Activity size={32} className={iconColor} />}
                        {status === 'ATTENTION' && <AlertCircle size={32} className={iconColor} />}
                    </div>
                </div>

                <div className="flex-1">
                    <h2 className="text-2xl font-bold text-primary tracking-tight">{message}</h2>
                    <p className="text-base text-secondary/90 mt-1 font-medium">{subMessage}</p>
                </div>
            </div>
        </motion.div>
    );
}
