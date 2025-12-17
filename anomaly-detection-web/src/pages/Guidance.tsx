import { useState, useEffect } from 'react';
import { Wind, Heart, Phone, Activity, Sparkles, Utensils, Clock, Droplets, Pill } from 'lucide-react';
import { BreathingExercise } from '../components/BreathingExercise';
import { LogModal } from '../components/LogModal';
import { useUser } from '../contexts/UserContext';
import { InsightCard } from '../components/InsightCard';
import { insightsEngine } from '../utils/insightsEngine';
import { achievementSystem } from '../utils/achievements';
import { motion } from 'framer-motion';

const container = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: {
            staggerChildren: 0.1
        }
    }
};

const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
};

export function Guidance() {
    const { addLog } = useUser();
    const [showBreathing, setShowBreathing] = useState(false);
    const [activeLogModal, setActiveLogModal] = useState<'exercise' | 'meal' | null>(null);
    const [breathingMode, setBreathingMode] = useState<'4-7-8' | 'box'>('4-7-8');
    const [insights, setInsights] = useState(insightsEngine.getInsights());

    useEffect(() => {
        // Update insights periodically
        const interval = setInterval(() => {
            setInsights(insightsEngine.getInsights());
        }, 60000); // Every minute

        return () => clearInterval(interval);
    }, []);

    const handleBreathingStart = (mode: '4-7-8' | 'box') => {
        setBreathingMode(mode);
        setShowBreathing(true);
    };

    const handleBreathingClose = () => {
        setShowBreathing(false);
        achievementSystem.incrementBreathingSessions();
    };

    const handleLogSave = (title: string, details: string) => {
        if (!activeLogModal) return;

        addLog({
            type: activeLogModal,
            title,
            details
        });

        // You could also trigger an achievement or toast here
        setActiveLogModal(null);
    };

    return (
        <motion.div
            variants={container}
            initial="hidden"
            animate="show"
            className="space-y-6 w-full"
        >
            {/* Hero Section */}
            <motion.div variants={item} className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-accent/20 to-primary/10 p-8 lg:p-10 shadow-lg border border-accent/10">
                <div className="absolute top-0 right-0 p-10 opacity-10">
                    <Sparkles size={120} className="lg:w-40 lg:h-40" />
                </div>
                <div className="relative z-10">
                    <h2 className="text-3xl lg:text-4xl font-bold text-primary mb-2">Wellness Hub</h2>
                    <p className="text-secondary text-lg lg:text-xl max-w-md">
                        Personalized guidance to improve your physical and mental well-being.
                    </p>
                </div>
            </motion.div>

            {/* AI Insights */}
            {insights.length > 0 && (
                <motion.div variants={item} className="space-y-3">
                    <h3 className="text-sm font-bold text-primary uppercase tracking-wide px-1 flex items-center gap-2">
                        <Activity size={14} className="text-accent" /> Health Insights
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {insights.slice(0, 3).map((insight) => (
                            <InsightCard key={insight.id} {...insight} />
                        ))}
                    </div>
                </motion.div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                    {/* Breathing Exercises */}
                    <motion.div variants={item} className="space-y-3">
                        <h3 className="text-sm font-bold text-primary uppercase tracking-wide px-1">Deep Breathing</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <motion.button
                                whileHover={{ scale: 1.02, backgroundColor: "rgba(255,255,255,0.03)" }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => handleBreathingStart('box')}
                                className="card-base p-5 bg-surface text-left group"
                            >
                                <div className="flex items-center gap-4 lg:gap-5">
                                    <div className="size-12 lg:size-14 rounded-2xl bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-all flex items-center justify-center shrink-0">
                                        <Wind size={24} className="lg:w-7 lg:h-7" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-primary lg:text-lg">Box Breathing</h3>
                                        <p className="text-sm lg:text-base text-secondary mt-1">Focus & Calm</p>
                                    </div>
                                </div>
                            </motion.button>

                            <motion.button
                                whileHover={{ scale: 1.02, backgroundColor: "rgba(255,255,255,0.03)" }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => handleBreathingStart('4-7-8')}
                                className="card-base p-5 bg-surface text-left group"
                            >
                                <div className="flex items-center gap-4 lg:gap-5">
                                    <div className="size-12 lg:size-14 rounded-2xl bg-success/10 text-success group-hover:bg-success group-hover:text-white transition-all flex items-center justify-center shrink-0">
                                        <Wind size={24} className="lg:w-7 lg:h-7" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-primary lg:text-lg">4-7-8 Technique</h3>
                                        <p className="text-sm lg:text-base text-secondary mt-1">Sleep & Relax</p>
                                    </div>
                                </div>
                            </motion.button>
                        </div>
                    </motion.div>


                    {/* Health Tips */}
                    <motion.div variants={item} className="space-y-3">
                        <h3 className="text-sm font-bold text-primary uppercase tracking-wide px-1">Daily Tips</h3>
                        <div className="grid grid-cols-1 gap-4">
                            {/* Heart Health */}
                            <motion.div whileHover={{ y: -2 }} className="card-base p-5 bg-surface hover:shadow-lg transition-all">
                                <div className="flex items-start gap-4">
                                    <div className="size-10 rounded-full bg-danger/10 text-danger flex items-center justify-center shrink-0 mt-1">
                                        <Heart size={20} />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-primary mb-2">Heart Health</h3>
                                        <p className="text-sm text-secondary leading-relaxed">
                                            Aim for 30 minutes of moderate activity. Taking a brisk walk during lunch can significantly improve cardiovascular health.
                                        </p>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Sleep */}
                            <motion.div whileHover={{ y: -2 }} className="card-base p-5 bg-surface hover:shadow-lg transition-all">
                                <div className="flex items-start gap-4">
                                    <div className="size-10 rounded-full bg-accent/10 text-accent flex items-center justify-center shrink-0 mt-1">
                                        <Sparkles size={20} />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-primary mb-2">Sleep Hygiene</h3>
                                        <p className="text-sm text-secondary leading-relaxed">
                                            Avoid screens 1 hour before bed. The blue light suppresses melatonin production, making it harder to fall asleep.
                                        </p>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Nutrition */}
                            <motion.div whileHover={{ y: -2 }} className="card-base p-5 bg-surface hover:shadow-lg transition-all">
                                <div className="flex items-start gap-4">
                                    <div className="size-10 rounded-full bg-success/10 text-success flex items-center justify-center shrink-0 mt-1">
                                        <Utensils size={20} />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-primary mb-2">Nutrition</h3>
                                        <p className="text-sm text-secondary leading-relaxed">
                                            Include more fiber in your diet. Whole grains, fruits, and vegetables can improve digestion and heart health.
                                        </p>
                                    </div>
                                </div>
                            </motion.div>

                            {/* Hydration */}
                            <motion.div whileHover={{ y: -2 }} className="card-base p-5 bg-surface hover:shadow-lg transition-all">
                                <div className="flex items-start gap-4">
                                    <div className="size-10 rounded-full bg-blue-500/10 text-blue-500 flex items-center justify-center shrink-0 mt-1">
                                        <Activity size={20} />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-primary mb-2">Hydration</h3>
                                        <p className="text-sm text-secondary leading-relaxed">
                                            Drink at least 8 glasses of water daily. Proper hydration helps maintain energy levels and brain function.
                                        </p>
                                    </div>
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                </div>

                <div className="space-y-6">
                    {/* Smart Schedule (Flagship Phase 3) */}
                    <motion.div variants={item} className="card-base p-6 bg-surface shadow-md">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-blue-500/10 text-blue-500 flex items-center justify-center">
                                <Clock size={20} />
                            </div>
                            <div>
                                <h3 className="font-bold text-primary">Smart Schedule</h3>
                                <p className="text-xs text-secondary">AI-optimized reminders</p>
                            </div>
                        </div>

                        <div className="space-y-4">
                            {/* Hydration */}
                            <div className="flex items-center justify-between p-4 rounded-2xl bg-background border border-white/5">
                                <div className="flex items-center gap-4">
                                    <div className="size-10 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center">
                                        <Droplets size={20} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-primary">Hydration Stats</h4>
                                        <p className="text-xs text-secondary">Next sip in 15 mins</p>
                                    </div>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" defaultChecked className="sr-only peer" />
                                    <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
                                </label>
                            </div>

                            {/* Medication */}
                            <div className="flex items-center justify-between p-4 rounded-2xl bg-background border border-white/5">
                                <div className="flex items-center gap-4">
                                    <div className="size-10 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center">
                                        <Pill size={20} />
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-primary">Medication</h4>
                                        <p className="text-xs text-secondary">Evening dose: 8:00 PM</p>
                                    </div>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" defaultChecked className="sr-only peer" />
                                    <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-500"></div>
                                </label>
                            </div>
                        </div>
                    </motion.div>

                    {/* Quick Actions */}
                    <motion.div variants={item} className="card-base p-6 bg-surface shadow-md">
                        <h3 className="font-bold text-primary mb-4">Quick Actions</h3>
                        <div className="space-y-3">
                            <button
                                onClick={() => handleBreathingStart('4-7-8')}
                                className="w-full flex items-center gap-3 bg-background hover:bg-surfaceHover border border-white/5 rounded-xl p-3 transition-colors text-left"
                            >
                                <div className="size-8 rounded-full bg-accent/10 text-accent flex items-center justify-center">
                                    <Wind size={16} />
                                </div>
                                <span className="text-sm font-medium text-primary">Quick Breath</span>
                            </button>
                            <button
                                onClick={() => setActiveLogModal('exercise')}
                                className="w-full flex items-center gap-3 bg-background hover:bg-surfaceHover border border-white/5 rounded-xl p-3 transition-colors text-left"
                            >
                                <div className="size-8 rounded-full bg-success/10 text-success flex items-center justify-center">
                                    <Activity size={16} />
                                </div>
                                <span className="text-sm font-medium text-primary">Log Exercise</span>
                            </button>
                            <button
                                onClick={() => setActiveLogModal('meal')}
                                className="w-full flex items-center gap-3 bg-background hover:bg-surfaceHover border border-white/5 rounded-xl p-3 transition-colors text-left"
                            >
                                <div className="size-8 rounded-full bg-warning/10 text-warning flex items-center justify-center">
                                    <Utensils size={16} />
                                </div>
                                <span className="text-sm font-medium text-primary">Log Meal</span>
                            </button>
                        </div>
                    </motion.div>

                    {/* Emergency Contact */}
                    <motion.div variants={item} className="card-base p-6 bg-gradient-to-br from-danger/5 to-surface border border-danger/10">
                        <div className="flex items-center gap-4 mb-4">
                            <div className="size-10 rounded-full bg-danger/10 text-danger flex items-center justify-center shrink-0">
                                <Phone size={20} />
                            </div>
                            <div>
                                <h3 className="font-bold text-primary">Emergency</h3>
                                <p className="text-xs text-secondary">SOS Helper</p>
                            </div>
                        </div>
                        <button className="w-full py-2.5 bg-danger text-white rounded-lg text-sm font-bold shadow-lg shadow-danger/20 hover:bg-danger/90 transition-all active:scale-95">
                            Call Emergency
                        </button>
                    </motion.div>
                </div>
            </div>

            {/* Breathing Exercise Modal */}
            <BreathingExercise
                isOpen={showBreathing}
                onClose={handleBreathingClose}
                mode={breathingMode}
            />

            {/* Logging Modal */}
            <LogModal
                isOpen={!!activeLogModal}
                onClose={() => setActiveLogModal(null)}
                type={activeLogModal}
                onSave={handleLogSave}
            />
        </motion.div>
    );
}
