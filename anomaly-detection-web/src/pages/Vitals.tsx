import { useState } from 'react';
import { Heart, Activity, Zap, Moon, TrendingUp, Calendar } from 'lucide-react';
import { RealTimeChart } from '../components/RealTimeChart';
import { InsightCard } from '../components/InsightCard';

type TimeRange = '1H' | '6H' | '1D' | '1W' | '1M';

export function Vitals() {
    const [selectedMetric, setSelectedMetric] = useState<'hr' | 'stress' | 'spo2' | 'sleep'>('hr');
    const [timeRange, setTimeRange] = useState<TimeRange>('1D');

    // Generate sample data based on time range
    const generateData = (points: number, base: number, variance: number) => {
        return Array.from({ length: points }, (_, i) =>
            base + Math.sin(i / 10) * variance + (Math.random() - 0.5) * variance * 0.5
        );
    };

    const getDataForMetric = () => {
        const points = timeRange === '1H' ? 60 : timeRange === '6H' ? 72 : timeRange === '1D' ? 96 : timeRange === '1W' ? 168 : 720;

        switch (selectedMetric) {
            case 'hr':
                return {
                    data: generateData(points, 72, 15),
                    labels: Array.from({ length: points }, (_, i) => `${i}`),
                    color: '#00C7BE',
                    min: 40,
                    max: 160,
                    normalMin: 60,
                    normalMax: 100,
                    unit: 'BPM',
                    title: 'Heart Rate',
                    avg: 72,
                    trend: '+2%'
                };
            case 'stress':
                return {
                    data: generateData(points, 45, 20),
                    labels: Array.from({ length: points }, (_, i) => `${i}`),
                    color: '#FFD60A',
                    min: 0,
                    max: 100,
                    normalMin: 0,
                    normalMax: 50,
                    unit: '%',
                    title: 'Stress Level',
                    avg: 45,
                    trend: '-8%'
                };
            case 'spo2':
                return {
                    data: generateData(points, 97, 2),
                    labels: Array.from({ length: points }, (_, i) => `${i}`),
                    color: '#30D158',
                    min: 90,
                    max: 100,
                    normalMin: 95,
                    normalMax: 100,
                    unit: '%',
                    title: 'Blood Oxygen',
                    avg: 97,
                    trend: '+0.5%'
                };
            case 'sleep':
                return {
                    data: generateData(points, 7.5, 1.5),
                    labels: Array.from({ length: points }, (_, i) => `${i}`),
                    color: '#00C7BE',
                    min: 0,
                    max: 12,
                    normalMin: 7,
                    normalMax: 9,
                    unit: 'hours',
                    title: 'Sleep Duration',
                    avg: 7.5,
                    trend: '+12%'
                };
        }
    };

    const chartData = getDataForMetric();

    const getInsightsForMetric = () => {
        switch (selectedMetric) {
            case 'hr':
                return [
                    {
                        type: 'achievement' as const,
                        severity: 'success' as const,
                        title: 'Excellent Recovery',
                        description: 'Your heart rate recovery after activity is in the top 10% for your age group.',
                        icon: 'trophy'
                    },
                    {
                        type: 'pattern' as const,
                        severity: 'info' as const,
                        title: 'Resting HR Trend',
                        description: 'Your resting heart rate has decreased by 2 BPM this week, indicating improved fitness.',
                        icon: 'trending'
                    },
                    {
                        type: 'recommendation' as const,
                        severity: 'info' as const,
                        title: 'Zone 2 Training',
                        description: 'Try to maintain 110-130 BPM for 30 minutes today to boost endurance.',
                        icon: 'activity'
                    },
                    {
                        type: 'correlation' as const,
                        severity: 'warning' as const,
                        title: 'Caffeine Impact',
                        description: 'High readings detected at 10 AM, correlating with your coffee intake.',
                        icon: 'clock'
                    }
                ];
            case 'stress':
                return [
                    {
                        type: 'pattern' as const,
                        severity: 'warning' as const,
                        title: 'Afternoon Spikes',
                        description: 'Stress levels consistently peak between 2 PM and 4 PM.',
                        icon: 'clock'
                    },
                    {
                        type: 'recommendation' as const,
                        severity: 'success' as const,
                        title: 'Breathing Break',
                        description: 'A 5-minute box breathing session now could reduce stress by 20%.',
                        icon: 'wind'
                    },
                    {
                        type: 'correlation' as const,
                        severity: 'info' as const,
                        title: 'Sleep Correlation',
                        description: 'Lower stress days follow nights with >7 hours of sleep.',
                        icon: 'moon'
                    },
                    {
                        type: 'achievement' as const,
                        severity: 'success' as const,
                        title: 'Calm Streak',
                        description: 'You maintained low stress levels for 4 consecutive hours this morning!',
                        icon: 'trophy'
                    }
                ];
            case 'spo2':
                return [
                    {
                        type: 'pattern' as const,
                        severity: 'success' as const,
                        title: 'Stable Oxygen Levels',
                        description: 'Your SpO2 remained above 95% throughout the night, which is excellent.',
                        icon: 'trending'
                    },
                    {
                        type: 'info' as const,
                        severity: 'info' as const,
                        title: 'Altitude Check',
                        description: 'Note: Slight variations are normal if you are currently at a higher altitude.',
                        icon: 'activity'
                    },
                    {
                        type: 'correlation' as const,
                        severity: 'info' as const,
                        title: 'Sleep Quality Link',
                        description: 'Consistent oxygen levels are contributing to your high sleep score.',
                        icon: 'moon'
                    },
                    {
                        type: 'recommendation' as const,
                        severity: 'warning' as const,
                        title: 'Deep Breathing',
                        description: 'Take deep breaths if you notice levels dropping below 94% during the day.',
                        icon: 'wind'
                    }
                ];
            case 'sleep':
                return [
                    {
                        type: 'achievement' as const,
                        severity: 'success' as const,
                        title: '8 Hour Goal Met',
                        description: 'You achieved your sleep duration goal 5 times this week.',
                        icon: 'trophy'
                    },
                    {
                        type: 'pattern' as const,
                        severity: 'info' as const,
                        title: 'Deep Sleep Consistency',
                        description: 'Deep sleep percentage is stable at 18%, well within the healthy range.',
                        icon: 'moon'
                    },
                    {
                        type: 'recommendation' as const,
                        severity: 'warning' as const,
                        title: 'Screen Time',
                        description: 'Screen usage 1 hour before bed may be delaying sleep onset.',
                        icon: 'clock'
                    },
                    {
                        type: 'correlation' as const,
                        severity: 'info' as const,
                        title: 'Activity Impact',
                        description: 'Days with >8k steps correlate with 15% more deep sleep.',
                        icon: 'activity'
                    }
                ];
            default:
                return [];
        }
    };

    const insights = getInsightsForMetric();

    return (
        <div className="space-y-6 w-full max-w-7xl mx-auto">
            {/* Header */}
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-primary/10 to-surface p-8 border border-white/5">
                <div className="relative z-10">
                    <h2 className="text-3xl font-bold text-primary mb-2">Vitals Dashboard</h2>
                    <p className="text-secondary text-lg">Detailed view of your health metrics</p>
                </div>
                <div className="absolute right-0 top-0 h-full w-1/3 bg-gradient-to-l from-accent/5 to-transparent" />
            </div>

            {/* Metric Selector */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
                <button
                    onClick={() => setSelectedMetric('hr')}
                    className={`relative overflow-hidden rounded-2xl p-5 lg:p-6 text-left border transition-all duration-300 hover:-translate-y-1 ${selectedMetric === 'hr'
                        ? 'bg-accent/10 border-accent/30 shadow-lg shadow-accent/10'
                        : 'bg-surface border-white/5 hover:bg-surfaceHover'
                        }`}
                >
                    <div className="flex items-center gap-3 lg:gap-5">
                        <div className={`size-12 lg:size-14 rounded-2xl flex items-center justify-center transition-colors ${selectedMetric === 'hr' ? 'bg-accent text-white shadow-lg shadow-accent/30' : 'bg-surface border border-white/10 text-secondary'
                            }`}>
                            <Heart size={20} className={`lg:w-7 lg:h-7 ${selectedMetric === 'hr' ? 'animate-pulse' : ''}`} />
                        </div>
                        <div>
                            <div className="text-xs lg:text-base font-semibold text-secondary uppercase tracking-wider">Heart Rate</div>
                            <div className="text-xl lg:text-4xl font-bold text-primary mt-0.5 tracking-tight">72 <span className="text-xs lg:text-sm text-secondary font-medium">BPM</span></div>
                        </div>
                    </div>
                </button>

                <button
                    onClick={() => setSelectedMetric('stress')}
                    className={`relative overflow-hidden rounded-2xl p-5 lg:p-6 text-left border transition-all duration-300 hover:-translate-y-1 ${selectedMetric === 'stress'
                        ? 'bg-warning/10 border-warning/30 shadow-lg shadow-warning/10'
                        : 'bg-surface border-white/5 hover:bg-surfaceHover'
                        }`}
                >
                    <div className="flex items-center gap-3 lg:gap-5">
                        <div className={`size-12 lg:size-14 rounded-2xl flex items-center justify-center transition-colors ${selectedMetric === 'stress' ? 'bg-warning text-white shadow-lg shadow-warning/30' : 'bg-surface border border-white/10 text-secondary'
                            }`}>
                            <Zap size={20} className="lg:w-7 lg:h-7" />
                        </div>
                        <div>
                            <div className="text-xs lg:text-base font-semibold text-secondary uppercase tracking-wider">Stress</div>
                            <div className="text-xl lg:text-4xl font-bold text-primary mt-0.5 tracking-tight">45 <span className="text-xs lg:text-sm text-secondary font-medium">%</span></div>
                        </div>
                    </div>
                </button>

                <button
                    onClick={() => setSelectedMetric('spo2')}
                    className={`relative overflow-hidden rounded-2xl p-5 lg:p-6 text-left border transition-all duration-300 hover:-translate-y-1 ${selectedMetric === 'spo2'
                        ? 'bg-success/10 border-success/30 shadow-lg shadow-success/10'
                        : 'bg-surface border-white/5 hover:bg-surfaceHover'
                        }`}
                >
                    <div className="flex items-center gap-3 lg:gap-5">
                        <div className={`size-12 lg:size-14 rounded-2xl flex items-center justify-center transition-colors ${selectedMetric === 'spo2' ? 'bg-success text-white shadow-lg shadow-success/30' : 'bg-surface border border-white/10 text-secondary'
                            }`}>
                            <Activity size={20} className="lg:w-7 lg:h-7" />
                        </div>
                        <div>
                            <div className="text-xs lg:text-base font-semibold text-secondary uppercase tracking-wider">SpO₂</div>
                            <div className="text-xl lg:text-4xl font-bold text-primary mt-0.5 tracking-tight">97 <span className="text-xs lg:text-sm text-secondary font-medium">%</span></div>
                        </div>
                    </div>
                </button>

                <button
                    onClick={() => setSelectedMetric('sleep')}
                    className={`relative overflow-hidden rounded-2xl p-5 lg:p-6 text-left border transition-all duration-300 hover:-translate-y-1 ${selectedMetric === 'sleep'
                        ? 'bg-info/10 border-info/30 shadow-lg shadow-info/10'
                        : 'bg-surface border-white/5 hover:bg-surfaceHover'
                        }`}
                >
                    <div className="flex items-center gap-3 lg:gap-5">
                        <div className={`size-12 lg:size-14 rounded-2xl flex items-center justify-center transition-colors ${selectedMetric === 'sleep' ? 'bg-info text-white shadow-lg shadow-info/30' : 'bg-surface border border-white/10 text-secondary'
                            }`}>
                            <Moon size={20} className="lg:w-7 lg:h-7" />
                        </div>
                        <div>
                            <div className="text-xs lg:text-base font-semibold text-secondary uppercase tracking-wider">Sleep</div>
                            <div className="text-xl lg:text-4xl font-bold text-primary mt-0.5 tracking-tight">7.5 <span className="text-xs lg:text-sm text-secondary font-medium">hrs</span></div>
                        </div>
                    </div>
                </button>
            </div>

            {/* Stats Summary */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg lg:text-xl font-bold text-primary">{chartData.title} Overview</h3>
                    <div className={`flex items-center gap-1 text-sm font-medium ${chartData.trend.startsWith('+') ? 'text-success' : 'text-danger'
                        }`}>
                        <TrendingUp size={16} />
                        {chartData.trend}
                    </div>
                </div>
                <div className="grid grid-cols-3 gap-4 lg:gap-8">
                    <div className="text-center">
                        <div className="text-2xl lg:text-4xl font-bold text-primary">{chartData.avg}</div>
                        <div className="text-sm text-secondary mt-1">Average</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl lg:text-4xl font-bold text-primary">{Math.max(...chartData.data).toFixed(0)}</div>
                        <div className="text-sm text-secondary mt-1">Peak</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl lg:text-4xl font-bold text-primary">{Math.min(...chartData.data).toFixed(0)}</div>
                        <div className="text-sm text-secondary mt-1">Lowest</div>
                    </div>
                </div>
            </div>

            {/* Time Range Selector */}
            <div className="card-base p-4 bg-surface">
                <div className="flex items-center gap-2 overflow-x-auto">
                    <Calendar size={16} className="text-secondary shrink-0" />
                    {(['1H', '6H', '1D', '1W', '1M'] as TimeRange[]).map((range) => (
                        <button
                            key={range}
                            onClick={() => setTimeRange(range)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors shrink-0 ${timeRange === range
                                ? 'bg-accent text-surface'
                                : 'bg-background text-secondary hover:text-primary'
                                }`}
                        >
                            {range}
                        </button>
                    ))}
                </div>
            </div>

            {/* Chart */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="h-64 lg:h-96 transition-all duration-300">
                    <RealTimeChart
                        label={chartData.title}
                        data={chartData.data}
                        labels={chartData.labels}
                        color={chartData.color}
                        min={chartData.min}
                        max={chartData.max}
                        normalMin={chartData.normalMin}
                        normalMax={chartData.normalMax}
                    />
                </div>
            </div>

            {/* AI Insights - Professional Grid */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="flex items-center gap-3 mb-6">
                    <div className="size-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                        <Zap size={20} />
                    </div>
                    <div>
                        <h3 className="text-lg lg:text-xl font-bold text-primary">AI Analysis</h3>
                        <p className="text-sm text-secondary">Intelligent insights based on your {selectedMetric === 'hr' ? 'heart rate' : selectedMetric} data</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 lg:gap-6">
                    {insights.map((insight, index) => (
                        <InsightCard
                            key={index}
                            type={insight.type as any}
                            severity={insight.severity as any}
                            title={insight.title}
                            description={insight.description}
                            icon={insight.icon}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
}
