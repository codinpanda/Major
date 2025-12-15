import { motion } from 'framer-motion';

interface ToggleProps {
    checked: boolean;
    onChange: (checked: boolean) => void;
}

export function Toggle({ checked, onChange }: ToggleProps) {
    return (
        <button
            onClick={() => onChange(!checked)}
            className={`w-12 h-7 rounded-full p-1 transition-colors duration-300 ${checked ? 'bg-accent' : 'bg-white/10'
                }`}
        >
            <motion.div
                className="w-5 h-5 bg-white rounded-full shadow-md"
                animate={{
                    x: checked ? 20 : 0
                }}
                transition={{
                    type: "spring",
                    stiffness: 500,
                    damping: 30
                }}
            />
        </button>
    );
}
