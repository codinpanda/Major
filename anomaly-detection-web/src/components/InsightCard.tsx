import { Clock, Moon, Trophy, Wind, TrendingUp, Activity } from 'lucide-react';

interface InsightCardProps {
    type: 'pattern' | 'correlation' | 'achievement' | 'recommendation';
    severity: 'info' | 'success' | 'warning';
    title: string;
    description: string;
    icon: string;
}

export function InsightCard({ type, severity, title, description, icon }: InsightCardProps) {
    const getIcon = () => {
        switch (icon) {
            case 'clock': return Clock;
            case 'moon': return Moon;
            case 'trophy': return Trophy;
            case 'wind': return Wind;
            case 'trending': return TrendingUp;
            default: return Activity;
        }
    };

    const Icon = getIcon();

    const getColors = () => {
        if (severity === 'success') return {
            bg: 'bg-success/5',
            border: 'border-success/10',
            iconBg: 'bg-success/10',
            iconText: 'text-success',
        };
        if (severity === 'warning') return {
            bg: 'bg-warning/5',
            border: 'border-warning/10',
            iconBg: 'bg-warning/10',
            iconText: 'text-warning',
        };
        return {
            bg: 'bg-accent/5',
            border: 'border-accent/10',
            iconBg: 'bg-accent/10',
            iconText: 'text-accent',
        };
    };

    const colors = getColors();

    const getTypeLabel = () => {
        switch (type) {
            case 'pattern': return 'Pattern';
            case 'correlation': return 'Insight';
            case 'achievement': return 'Achievement';
            case 'recommendation': return 'Tip';
        }
    };

    return (
        <div className={`flex items-start gap-3 p-4 rounded-xl border ${colors.bg} ${colors.border} animate-fade-in`}>
            <div className={`size-10 rounded-full ${colors.iconBg} ${colors.iconText} flex items-center justify-center shrink-0 mt-0.5`}>
                <Icon size={20} />
            </div>
            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                    <span className={`text-xs font-medium ${colors.iconText} uppercase tracking-wide`}>
                        {getTypeLabel()}
                    </span>
                </div>
                <div className="text-sm font-medium text-primary mb-1">{title}</div>
                <div className="text-xs text-secondary">{description}</div>
            </div>
        </div>
    );
}
