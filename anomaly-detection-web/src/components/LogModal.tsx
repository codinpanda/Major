import { motion, AnimatePresence } from 'framer-motion';
import { X, Save } from 'lucide-react';
import { useState, useEffect } from 'react';

interface LogModalProps {
    isOpen: boolean;
    onClose: () => void;
    type: 'exercise' | 'meal' | null;
    onSave: (title: string, details: string) => void;
}

export function LogModal({ isOpen, onClose, type, onSave }: LogModalProps) {
    const [title, setTitle] = useState("");
    const [details, setDetails] = useState("");

    useEffect(() => {
        if (isOpen) {
            setTitle("");
            setDetails("");
        }
    }, [isOpen]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (title.trim()) {
            onSave(title, details);
            onClose();
        }
    };

    if (!isOpen || !type) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={onClose}
                    className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                />

                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 20 }}
                    className="relative w-full max-w-md bg-[#161b22] border border-[#30363d] rounded-2xl shadow-xl overflow-hidden"
                >
                    <div className="flex items-center justify-between p-4 border-b border-[#30363d]">
                        <h3 className="text-lg font-bold text-white capitalize">
                            Log {type}
                        </h3>
                        <button onClick={onClose} className="p-2 hover:bg-[#30363d] rounded-lg transition-colors text-gray-400 hover:text-white">
                            <X size={20} />
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="p-4 space-y-4">
                        <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wide">
                                {type === 'exercise' ? 'Activity Name' : 'Meal Name'}
                            </label>
                            <input
                                type="text"
                                value={title}
                                onChange={(e) => setTitle(e.target.value)}
                                placeholder={type === 'exercise' ? "e.g., Morning Run" : "e.g., Oatmeal & Fruit"}
                                className="w-full bg-[#0d1117] border border-[#30363d] rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500 transition-colors"
                                autoFocus
                            />
                        </div>

                        <div>
                            <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wide">
                                {type === 'exercise' ? 'Duration / Intensity' : 'Calories / Portion'}
                            </label>
                            <input
                                type="text"
                                value={details}
                                onChange={(e) => setDetails(e.target.value)}
                                placeholder={type === 'exercise' ? "e.g., 30 mins, Moderate" : "e.g., 350 kcal, Medium Bowl"}
                                className="w-full bg-[#0d1117] border border-[#30363d] rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500 transition-colors"
                            />
                        </div>

                        <div className="pt-2">
                            <button
                                type="submit"
                                disabled={!title.trim()}
                                className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                            >
                                <Save size={18} />
                                Save Log
                            </button>
                        </div>
                    </form>
                </motion.div>
            </div>
        </AnimatePresence>
    );
}
