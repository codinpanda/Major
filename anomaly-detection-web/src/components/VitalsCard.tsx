import clsx from 'clsx';

interface VitalsCardProps {
    title: string;
    value: string;
    unit: string;
    status?: 'normal' | 'warning' | 'critical';
    icon?: React.ElementType;
}

export function VitalsCard({ title, value, unit, status = 'normal', icon: Icon }: VitalsCardProps) {

    const statusMap = {
        normal: { text: 'Looks good', color: 'text-success', bg: 'from-success/10 to-transparent border-success/10' },
        warning: { text: 'Higher than usual', color: 'text-warning', bg: 'from-warning/10 to-transparent border-warning/10' },
        critical: { text: 'Needs attention', color: 'text-danger', bg: 'from-danger/10 to-transparent border-danger/10' },
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
                    <div className="text-[10px] sm:text-xs lg:text-base text-secondary font-semibold uppercase tracking-wide mb-1">
                        {title}
                    </div>
                    <div className="flex items-baseline gap-1.5 sm:gap-2 lg:gap-2.5">
                        <span className="text-3xl sm:text-4xl lg:text-4xl font-bold text-primary tracking-tight">
                            {value}
                        </span>
                        <span className="text-xs sm:text-sm lg:text-sm text-secondary font-medium">{unit}</span>
                    </div>
                    <div className={clsx("text-xs sm:text-sm lg:text-base font-medium mt-0.5 sm:mt-1 lg:mt-1.5", s.color)}>
                        {s.text}
                    </div>
                </div>
            </div>
        </div>
    );
}
