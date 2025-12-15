import { Calendar, TrendingUp, Activity, Utensils } from 'lucide-react';
import { useUser } from '../contexts/UserContext';

export function History() {
    const { user } = useUser();

    // Sort logs by timestamp (newest first)
    const sortedLogs = [...user.logs].sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );

    return (
        <div className="space-y-4 w-full">
            <div className="card-base p-6 bg-surface">
                <h2 className="text-xl font-bold text-primary">Your Health History</h2>
                <p className="text-secondary text-sm mt-2">Track your progress over time</p>
            </div>

            {/* Today's Summary */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="flex items-center gap-3 mb-4">
                    <div className="size-10 rounded-full bg-accent/10 text-accent flex items-center justify-center">
                        <Calendar size={20} />
                    </div>
                    <div>
                        <h3 className="font-bold text-primary">Today</h3>
                        <p className="text-xs text-secondary">December 14, 2025</p>
                    </div>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
                    <div className="bg-background rounded-xl p-3 lg:p-4">
                        <div className="text-xs lg:text-sm text-secondary uppercase tracking-wide">Avg Heart Rate</div>
                        <div className="text-2xl lg:text-4xl font-bold text-primary mt-1 tracking-tight">72 <span className="text-sm lg:text-base text-secondary font-medium">BPM</span></div>
                    </div>
                    <div className="bg-background rounded-xl p-3 lg:p-4">
                        <div className="text-xs lg:text-sm text-secondary uppercase tracking-wide">Stress Level</div>
                        <div className="text-2xl lg:text-4xl font-bold text-primary mt-1 tracking-tight">42 <span className="text-sm lg:text-base text-secondary font-medium">%</span></div>
                    </div>
                </div>
            </div>

            {/* Weekly Trend */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="flex items-center gap-3 mb-4">
                    <div className="size-10 rounded-full bg-success/10 text-success flex items-center justify-center">
                        <TrendingUp size={20} />
                    </div>
                    <div>
                        <h3 className="font-bold text-primary">This Week</h3>
                        <p className="text-xs text-secondary">Your metrics are improving</p>
                    </div>
                </div>
                <div className="space-y-2">
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-secondary">Average heart rate</span>
                        <span className="text-sm font-medium text-success">↓ 3 BPM</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-secondary">Stress levels</span>
                        <span className="text-sm font-medium text-success">↓ 8%</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-secondary">Sleep quality</span>
                        <span className="text-sm font-medium text-success">↑ 12%</span>
                    </div>
                </div>
            </div>

            {/* Activity Log */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="flex items-center gap-3 mb-4">
                    <div className="size-10 rounded-full bg-warning/10 text-warning flex items-center justify-center">
                        <Activity size={20} />
                    </div>
                    <div>
                        <h3 className="font-bold text-primary">Recent Activity</h3>
                        <p className="text-xs text-secondary">Manual logs & device events</p>
                    </div>
                </div>
                <div className="space-y-3">
                    {sortedLogs.length === 0 ? (
                        <div className="text-center py-4 text-secondary text-sm">
                            No recent activities logged.
                        </div>
                    ) : (
                        sortedLogs.map((log) => (
                            <div key={log.id} className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
                                <div className="flex items-center gap-3">
                                    <div className={`size-8 rounded-full flex items-center justify-center ${log.type === 'meal' ? 'bg-success/10 text-success' : 'bg-blue-500/10 text-blue-500'
                                        }`}>
                                        {log.type === 'meal' ? <Utensils size={14} /> : <Activity size={14} />}
                                    </div>
                                    <div>
                                        <div className="text-sm text-primary font-medium capitalize">{log.title}</div>
                                        <div className="text-xs text-secondary">
                                            {new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </div>
                                    </div>
                                </div>
                                <div className="text-sm text-secondary">{log.details}</div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}
