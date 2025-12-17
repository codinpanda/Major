import clsx from 'clsx';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface VitalsCardProps {
    title: string;
    value: string;
    unit: string;
    status?: 'normal' | 'warning' | 'critical';
    icon?: React.ElementType;
    trend?: 'up' | 'down' | 'stable'; // New Prop
}

export function VitalsCard({ title, value, unit, status = 'normal', icon: Icon, trend = 'stable' }: VitalsCardProps) {

    const statusMap = {
        normal: { text: 'Normal', color: 'text-success', bg: 'from-success/10 to-transparent border-success/10' },
        warning: { text: 'Elevated', color: 'text-warning', bg: 'from-warning/10 to-transparent border-warning/10' },
        critical: { text: 'Critical', color: 'text-danger', bg: 'from-danger/10 to-transparent border-danger/10' },
    };

    const s = statusMap[status];

    return (
        <div className={clsx(
            "card-base p-4 sm:p-5 lg:p-6 transition-all duration-300 group hover:-translate-y-1 hover:shadow-lg relative overflow-hidden",
            "bg-gradient-to-br border",
            s.bg
        )}>
            {/* Glossy Overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

            <div className="relative flex items-center gap-3 sm:gap-4 lg:gap-5">
                {/* Icon Left - Responsive Size */}
                {Icon && (
                    <div className={clsx(
                        "size-10 sm:size-12 lg:size-14 rounded-2xl flex items-center justify-center shrink-0 transition-transform group-hover:scale-110",
                        status === 'normal' ? "bg-success/20 text-success" :
                            status === 'warning' ? "bg-warning/20 text-warning" :
                                "bg-danger/20 text-danger"
                    )}>
                        <Icon size={20} className="sm:w-6 sm:h-6 lg:w-7 lg:h-7" />
                    </div>
                )}

                {/* Content - Responsive Typography */}
                <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start">
                        <div className="text-[10px] sm:text-xs lg:text-base text-secondary font-semibold uppercase tracking-wide mb-1">
                            {title}
                        </div>
                        {/* Trend Indicator */}
                        <div className={clsx("flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-surface/50 border border-white/5", s.color)}>
                            {trend === 'up' && <TrendingUp size={12} />}
                            {trend === 'down' && <TrendingDown size={12} />}
                            {trend === 'stable' && <Minus size={12} />}
                            <span>{trend === 'up' ? '+2%' : trend === 'down' ? '-2%' : '0%'}</span>
                        </div>
                    </div>

                    <div className="flex items-baseline gap-1.5 sm:gap-2 lg:gap-2.5">
                        <span className="text-3xl sm:text-4xl lg:text-4xl font-bold text-primary tracking-tight">
                            {value}
                        </span>
                        <span className="text-xs sm:text-sm lg:text-sm text-secondary font-medium">{unit}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
