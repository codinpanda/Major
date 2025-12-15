interface AchievementBadgeProps {
    title: string;
    description: string;
    icon: string;
    unlocked: boolean;
    progress: number;
    unlockedAt?: number;
}

export function AchievementBadge({ title, description, icon, unlocked, progress, unlockedAt }: AchievementBadgeProps) {
    return (
        <div className={`card-base p-4 transition-all ${unlocked ? 'bg-accent/10 border-accent/30' : 'bg-surface opacity-60'
            }`}>
            <div className="flex items-start gap-3">
                {/* Icon */}
                <div className={`text-4xl ${unlocked ? 'animate-pulse-slow' : 'grayscale'}`}>
                    {icon}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                    <h3 className="font-bold text-primary text-sm">{title}</h3>
                    <p className="text-xs text-secondary mt-1">{description}</p>

                    {/* Progress Bar */}
                    {!unlocked && (
                        <div className="mt-2">
                            <div className="h-1.5 bg-background rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-accent transition-all duration-500"
                                    style={{ width: `${progress}%` }}
                                />
                            </div>
                            <div className="text-xs text-secondary mt-1">{Math.round(progress)}%</div>
                        </div>
                    )}

                    {/* Unlocked Date */}
                    {unlocked && unlockedAt && (
                        <div className="text-xs text-accent mt-1">
                            Unlocked {new Date(unlockedAt).toLocaleDateString()}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
