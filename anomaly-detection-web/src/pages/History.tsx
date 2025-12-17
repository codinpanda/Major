import { useState } from 'react';
import { TrendingUp, Activity, Utensils, BarChart3, Clock } from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

export function History() {
    const { user } = useUser();
    const [activeTab, setActiveTab] = useState<'daily' | 'weekly' | 'monthly'>('daily');

    // Sort logs by timestamp (newest first)
    const sortedLogs = [...user.logs].sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );

    // Mock Data for Charts based on Tab
    const chartLabels = activeTab === 'daily'
        ? ['6AM', '9AM', '12PM', '3PM', '6PM', '9PM']
        : activeTab === 'weekly'
            ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            : ['Week 1', 'Week 2', 'Week 3', 'Week 4'];

    const chartData = {
        labels: chartLabels,
        datasets: [
            {
                label: 'Stress Level',
                data: activeTab === 'daily'
                    ? [20, 35, 45, 60, 40, 25]
                    : activeTab === 'weekly'
                        ? [30, 45, 25, 60, 55, 30, 20]
                        : [35, 40, 45, 30],
                borderColor: '#fb7185', // Soft Rose
                backgroundColor: (context: any) => {
                    const ctx = context.chart.ctx;
                    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
                    gradient.addColorStop(0, 'rgba(251, 113, 133, 0.4)');
                    gradient.addColorStop(1, 'rgba(251, 113, 133, 0.0)');
                    return gradient;
                },
                tension: 0.5, // Smoother curve
                fill: true,
                pointBackgroundColor: '#fb7185',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#fb7185',
                pointRadius: 4,
                pointHoverRadius: 6,
            },
            {
                label: 'Heart Rate Avg',
                data: activeTab === 'daily'
                    ? [65, 72, 78, 85, 75, 68]
                    : activeTab === 'weekly'
                        ? [70, 75, 72, 80, 78, 70, 68]
                        : [72, 74, 73, 70],
                borderColor: '#34d399', // Soft Emerald
                backgroundColor: (context: any) => {
                    const ctx = context.chart.ctx;
                    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
                    gradient.addColorStop(0, 'rgba(52, 211, 153, 0.4)');
                    gradient.addColorStop(1, 'rgba(52, 211, 153, 0.0)');
                    return gradient;
                },
                tension: 0.5,
                fill: true,
                pointBackgroundColor: '#34d399',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#34d399',
                pointRadius: 4,
                pointHoverRadius: 6,
            }
        ],
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        plugins: {
            legend: {
                position: 'top' as const,
                align: 'end' as const,
                labels: {
                    color: '#9ca3af',
                    usePointStyle: true,
                    boxWidth: 8,
                    padding: 20,
                    font: { size: 12, family: "'Inter', sans-serif" }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)', // Dark background
                titleColor: '#f9fafb',
                bodyColor: '#e5e7eb',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 12,
                boxPadding: 4,
                usePointStyle: true,
                callbacks: {
                    label: function (context: any) {
                        let label = context.dataset.label || '';
                        if (label) {
                            label += ': ';
                        }
                        if (context.parsed.y !== null) {
                            label += context.parsed.y + (context.datasetIndex === 0 ? '%' : ' BPM');
                        }
                        return label;
                    }
                }
            },
            title: { display: false },
        },
        scales: {
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' }, // Very subtle grid
                ticks: { color: '#6b7280', font: { size: 11 } },
                suggestedMin: 0,
            },
            x: {
                grid: { display: false },
                ticks: { color: '#6b7280', font: { size: 11 } }
            }
        }
    };

    return (
        <div className="space-y-6 w-full">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-primary">Analytics & History</h2>
                    <p className="text-secondary text-sm mt-1">Deep dive into your health trends</p>
                </div>

                {/* Time Range Tabs */}
                <div className="flex bg-surface border border-white/5 p-1 rounded-xl w-full sm:w-auto">
                    {(['daily', 'weekly', 'monthly'] as const).map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`flex-1 sm:flex-none px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab
                                ? 'bg-primary text-background shadow-lg'
                                : 'text-secondary hover:text-primary hover:bg-white/5'
                                } capitalize`}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
            </div>

            {/* Trends Chart */}
            <div className="card-base p-6 bg-surface">
                <div className="flex items-center gap-3 mb-6">
                    <div className="size-10 rounded-xl bg-accent/10 text-accent flex items-center justify-center">
                        <BarChart3 size={20} />
                    </div>
                    <div>
                        <h3 className="font-bold text-primary">Health Trends</h3>
                        <p className="text-xs text-secondary">Stress vs Heart Rate Correlation</p>
                    </div>
                </div>
                <div className="h-[300px] w-full">
                    <Line options={chartOptions} data={chartData} />
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Weekly Summary */}
                <div className="card-base p-6 bg-surface">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="size-10 rounded-xl bg-success/10 text-success flex items-center justify-center">
                            <TrendingUp size={20} />
                        </div>
                        <div>
                            <h3 className="font-bold text-primary">Performance</h3>
                            <p className="text-xs text-secondary">Weekly Average</p>
                        </div>
                    </div>
                    <div className="space-y-4">
                        <div className="flex justify-between items-center p-3 bg-background rounded-xl border border-white/5">
                            <span className="text-sm text-secondary">Avg Heart Rate</span>
                            <div className="text-right">
                                <div className="text-lg font-bold text-primary">72 BPM</div>
                                <div className="text-xs text-success">↓ 3% vs last week</div>
                            </div>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-background rounded-xl border border-white/5">
                            <span className="text-sm text-secondary">Avg Stress Score</span>
                            <div className="text-right">
                                <div className="text-lg font-bold text-primary">42/100</div>
                                <div className="text-xs text-success">↓ 8% vs last week</div>
                            </div>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-background rounded-xl border border-white/5">
                            <span className="text-sm text-secondary">Sleep Duration</span>
                            <div className="text-right">
                                <div className="text-lg font-bold text-primary">7h 12m</div>
                                <div className="text-xs text-success">↑ 15m vs last week</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Activity Log */}
                <div className="card-base p-6 bg-surface">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="size-10 rounded-xl bg-warning/10 text-warning flex items-center justify-center">
                            <Clock size={20} />
                        </div>
                        <div>
                            <h3 className="font-bold text-primary">Recent Timeline</h3>
                            <p className="text-xs text-secondary">Events & Alerts</p>
                        </div>
                    </div>
                    <div className="space-y-3 max-h-[250px] overflow-y-auto pr-2 custom-scrollbar">
                        {sortedLogs.length === 0 ? (
                            <div className="text-center py-4 text-secondary text-sm">
                                No recent activities logged.
                            </div>
                        ) : (
                            sortedLogs.map((log) => (
                                <div key={log.id} className="flex gap-4 items-start p-3 hover:bg-white/5 rounded-xl transition-colors border border-transparent hover:border-white/5">
                                    <div className={`mt-1 size-8 rounded-full flex items-center justify-center shrink-0 ${log.type === 'meal' ? 'bg-success/10 text-success' : 'bg-blue-500/10 text-blue-500'
                                        }`}>
                                        {log.type === 'meal' ? <Utensils size={14} /> : <Activity size={14} />}
                                    </div>
                                    <div>
                                        <div className="text-sm text-primary font-bold capitalize">{log.title}</div>
                                        <div className="text-xs text-secondary mt-0.5">{log.details}</div>
                                        <div className="text-[10px] text-secondary/60 mt-2 font-mono">
                                            {new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
