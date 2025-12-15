export function SkeletonLoader({ className = '' }: { className?: string }) {
    return (
        <div className={`animate-pulse ${className}`}>
            <div className="h-full w-full bg-surfaceHover rounded-2xl"></div>
        </div>
    );
}

export function CardSkeleton() {
    return (
        <div className="card-base p-5 bg-surface animate-pulse">
            <div className="flex items-center gap-4">
                <div className="size-12 rounded-full bg-surfaceHover"></div>
                <div className="flex-1 space-y-2">
                    <div className="h-3 bg-surfaceHover rounded w-1/3"></div>
                    <div className="h-6 bg-surfaceHover rounded w-1/2"></div>
                    <div className="h-3 bg-surfaceHover rounded w-1/4"></div>
                </div>
            </div>
        </div>
    );
}

export function ChartSkeleton() {
    return (
        <div className="card-base p-5 bg-surface animate-pulse">
            <div className="h-4 bg-surfaceHover rounded w-1/4 mb-4"></div>
            <div className="h-64 bg-surfaceHover rounded"></div>
        </div>
    );
}
