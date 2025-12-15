import { AlertCircle, RefreshCw } from 'lucide-react';

interface EmptyStateProps {
    title: string;
    description: string;
    icon?: React.ElementType;
    action?: {
        label: string;
        onClick: () => void;
    };
}

export function EmptyState({ title, description, icon: Icon = AlertCircle, action }: EmptyStateProps) {
    return (
        <div className="card-base p-12 bg-surface text-center">
            <div className="flex flex-col items-center gap-4">
                <div className="size-16 rounded-full bg-secondary/10 flex items-center justify-center">
                    <Icon size={32} className="text-secondary" />
                </div>
                <div>
                    <h3 className="text-lg font-bold text-primary mb-2">{title}</h3>
                    <p className="text-sm text-secondary max-w-sm mx-auto">{description}</p>
                </div>
                {action && (
                    <button
                        onClick={action.onClick}
                        className="mt-2 px-6 py-2.5 bg-accent text-surface rounded-xl font-medium hover:bg-accentDark transition-colors"
                    >
                        {action.label}
                    </button>
                )}
            </div>
        </div>
    );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
    return (
        <div className="card-base p-12 bg-surface text-center border border-danger/20">
            <div className="flex flex-col items-center gap-4">
                <div className="size-16 rounded-full bg-danger/10 flex items-center justify-center">
                    <AlertCircle size={32} className="text-danger" />
                </div>
                <div>
                    <h3 className="text-lg font-bold text-primary mb-2">Something went wrong</h3>
                    <p className="text-sm text-secondary max-w-sm mx-auto">{message}</p>
                </div>
                <button
                    onClick={onRetry}
                    className="mt-2 px-6 py-2.5 bg-accent text-surface rounded-xl font-medium hover:bg-accentDark transition-colors flex items-center gap-2"
                >
                    <RefreshCw size={18} />
                    Try Again
                </button>
            </div>
        </div>
    );
}
