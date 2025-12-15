import { Calendar, TrendingUp, Activity } from 'lucide-react';

export function History() {
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
                    <h3 className="font-bold text-primary">Recent Activity</h3>
                </div>
                <div className="space-y-3">
                    <div className="flex justify-between items-center py-2 border-b border-white/5">
                        <div>
                            <div className="text-sm text-primary font-medium">Morning walk</div>
                            <div className="text-xs text-secondary">8:30 AM</div>
                        </div>
                        <div className="text-sm text-secondary">32 min</div>
                    </div>
                    <div className="flex justify-between items-center py-2 border-b border-white/5">
                        <div>
                            <div className="text-sm text-primary font-medium">Breathing exercise</div>
                            <div className="text-xs text-secondary">2:15 PM</div>
                        </div>
                        <div className="text-sm text-secondary">5 min</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
