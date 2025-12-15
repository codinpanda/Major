import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCcw } from 'lucide-react';

interface Props {
    children?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-[#0d1117] flex items-center justify-center p-4">
                    <div className="max-w-md w-full bg-[#161b22] border border-red-500/20 rounded-2xl p-8 text-center shadow-2xl">
                        <div className="size-16 bg-red-500/10 text-red-500 rounded-full flex items-center justify-center mx-auto mb-6">
                            <AlertTriangle size={32} />
                        </div>
                        <h1 className="text-2xl font-bold text-white mb-2">System Encountered an Error</h1>
                        <p className="text-gray-400 mb-6">
                            The monitoring dashboard encountered an unexpected state.
                            Our engineering team has been notified.
                        </p>

                        <div className="bg-black/30 rounded-lg p-4 mb-6 text-left">
                            <code className="text-xs text-red-300 font-mono break-all">
                                {this.state.error?.message || "Unknown Error"}
                            </code>
                        </div>

                        <button
                            onClick={() => window.location.reload()}
                            className="flex items-center justify-center gap-2 w-full bg-red-600 hover:bg-red-700 text-white font-medium py-3 rounded-xl transition-colors"
                        >
                            <RefreshCcw size={18} />
                            Reboot System
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
