import { useState } from 'react';
import { Moon, TrendingUp, Calendar } from 'lucide-react';
import { RealTimeChart } from '../components/RealTimeChart';

type TimeRange = '1W' | '2W' | '1M' | '3M';

interface SleepData {
    date: string;
    duration: number; // hours
    quality: number; // 0-100
    deepSleep: number; // hours
    lightSleep: number; // hours
    awakenings: number;
}

export function SleepQuality() {
    const [timeRange, setTimeRange] = useState<TimeRange>('1W');

    // Generate mock sleep data
    const generateSleepData = (days: number): SleepData[] => {
        const data: SleepData[] = [];
        const today = new Date();

        for (let i = days - 1; i >= 0; i--) {
            const date = new Date(today);
            date.setDate(date.getDate() - i);

            const duration = 6 + Math.random() * 3; // 6-9 hours
            const quality = 60 + Math.random() * 35; // 60-95%
            const deepSleep = duration * (0.15 + Math.random() * 0.15); // 15-30% of sleep
            const lightSleep = duration - deepSleep - 0.5;
            const awakenings = Math.floor(Math.random() * 4);

            data.push({
                date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                duration,
                quality,
                deepSleep,
                lightSleep,
                awakenings,
            });
        }

        return data;
    };

    const getDaysForRange = (): number => {
        switch (timeRange) {
            case '1W': return 7;
            case '2W': return 14;
            case '1M': return 30;
            case '3M': return 90;
        }
    };

    const sleepData = generateSleepData(getDaysForRange());
    const avgDuration = sleepData.reduce((sum, d) => sum + d.duration, 0) / sleepData.length;
    const avgQuality = sleepData.reduce((sum, d) => sum + d.quality, 0) / sleepData.length;
    const avgDeepSleep = sleepData.reduce((sum, d) => sum + d.deepSleep, 0) / sleepData.length;

    const getQualityColor = (quality: number): string => {
        if (quality >= 80) return 'text-success';
        if (quality >= 60) return 'text-warning';
        return 'text-danger';
    };

    const getQualityLabel = (quality: number): string => {
        if (quality >= 80) return 'Excellent';
        if (quality >= 60) return 'Good';
        return 'Needs Improvement';
    };

    return (
        <div className="space-y-4 w-full">
            {/* Header */}
            <div className="card-base p-6 bg-surface">
                <div className="flex items-center gap-3">
                    <div className="size-12 rounded-full bg-accent/10 text-accent flex items-center justify-center">
                        <Moon size={24} />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-primary">Sleep Quality</h2>
                        <p className="text-secondary text-sm mt-1">Track your sleep patterns and quality</p>
                    </div>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 lg:gap-6">
                <div className="card-base p-4 lg:p-6 bg-surface text-center">
                    <div className="text-2xl lg:text-4xl font-bold text-primary tracking-tight">{avgDuration.toFixed(1)}</div>
                    <div className="text-xs lg:text-sm text-secondary mt-1">Avg Duration</div>
                    <div className="text-xs lg:text-sm text-accent mt-1">hours</div>
                </div>
                <div className="card-base p-4 lg:p-6 bg-surface text-center">
                    <div className={`text-2xl lg:text-4xl font-bold tracking-tight ${getQualityColor(avgQuality)}`}>
                        {avgQuality.toFixed(0)}%
                    </div>
                    <div className="text-xs lg:text-sm text-secondary mt-1">Avg Quality</div>
                    <div className={`text-xs lg:text-sm mt-1 ${getQualityColor(avgQuality)}`}>
                        {getQualityLabel(avgQuality)}
                    </div>
                </div>
                <div className="card-base p-4 lg:p-6 bg-surface text-center">
                    <div className="text-2xl lg:text-4xl font-bold text-primary tracking-tight">{avgDeepSleep.toFixed(1)}</div>
                    <div className="text-xs lg:text-sm text-secondary mt-1">Deep Sleep</div>
                    <div className="text-xs lg:text-sm text-accent mt-1">hours</div>
                </div>
            </div>

            {/* Time Range Selector */}
            <div className="card-base p-4 bg-surface">
                <div className="flex items-center gap-2 overflow-x-auto">
                    <Calendar size={16} className="text-secondary shrink-0" />
                    {(['1W', '2W', '1M', '3M'] as TimeRange[]).map((range) => (
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

            {/* Sleep Duration Chart */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <h3 className="font-bold text-primary mb-4 lg:text-lg">Sleep Duration</h3>
                <div className="h-64 lg:h-96 transition-all duration-300">
                    <RealTimeChart
                        label="Sleep"
                        data={sleepData.map(d => d.duration)}
                        labels={sleepData.map(d => d.date)}
                        color="#00C7BE"
                        min={0}
                        max={12}
                        normalMin={7}
                        normalMax={9}
                        unit=" hrs"
                    />
                </div>
            </div>

            {/* Sleep Quality Chart */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <h3 className="font-bold text-primary mb-4 lg:text-lg">Sleep Quality Score</h3>
                <div className="h-64 lg:h-96 transition-all duration-300">
                    <RealTimeChart
                        label="Quality"
                        data={sleepData.map(d => d.quality)}
                        labels={sleepData.map(d => d.date)}
                        color="#30D158"
                        min={0}
                        max={100}
                        normalMin={70}
                        normalMax={100}
                        unit="%"
                    />
                </div>
            </div>

            {/* Sleep Stages */}
            <div className="card-base p-5 bg-surface">
                <h3 className="font-bold text-primary mb-4">Sleep Stages</h3>
                <div className="space-y-3">
                    {sleepData.slice(-7).reverse().map((day, index) => (
                        <div key={index} className="space-y-2">
                            <div className="flex items-center justify-between text-sm">
                                <span className="text-secondary">{day.date}</span>
                                <span className="text-primary font-medium">{day.duration.toFixed(1)} hrs</span>
                            </div>
                            <div className="flex gap-1 h-2 rounded-full overflow-hidden">
                                <div
                                    className="bg-accent"
                                    style={{ width: `${(day.deepSleep / day.duration) * 100}%` }}
                                    title={`Deep: ${day.deepSleep.toFixed(1)}h`}
                                />
                                <div
                                    className="bg-accent/40"
                                    style={{ width: `${(day.lightSleep / day.duration) * 100}%` }}
                                    title={`Light: ${day.lightSleep.toFixed(1)}h`}
                                />
                                <div
                                    className="bg-warning/60"
                                    style={{ width: `${((day.duration - day.deepSleep - day.lightSleep) / day.duration) * 100}%` }}
                                    title="Awake"
                                />
                            </div>
                            <div className="flex items-center gap-4 text-xs text-secondary">
                                <div className="flex items-center gap-1">
                                    <div className="size-2 rounded-full bg-accent"></div>
                                    Deep: {day.deepSleep.toFixed(1)}h
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="size-2 rounded-full bg-accent/40"></div>
                                    Light: {day.lightSleep.toFixed(1)}h
                                </div>
                                <div className="flex items-center gap-1">
                                    <div className="size-2 rounded-full bg-warning/60"></div>
                                    Awake: {day.awakenings}x
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Sleep Insights */}
            <div className="card-base p-5 bg-surface">
                <h3 className="font-bold text-primary mb-3">Sleep Insights</h3>
                <div className="space-y-3">
                    {avgDuration >= 7 && (
                        <div className="flex items-start gap-3 p-3 bg-success/5 rounded-xl border border-success/10">
                            <div className="size-8 rounded-full bg-success/10 text-success flex items-center justify-center shrink-0 mt-0.5">
                                <TrendingUp size={16} />
                            </div>
                            <div>
                                <div className="text-sm font-medium text-primary">Great Sleep Duration!</div>
                                <div className="text-xs text-secondary mt-1">
                                    You're averaging {avgDuration.toFixed(1)} hours of sleep. Keep maintaining this healthy habit!
                                </div>
                            </div>
                        </div>
                    )}

                    {avgQuality >= 80 && (
                        <div className="flex items-start gap-3 p-3 bg-success/5 rounded-xl border border-success/10">
                            <div className="size-8 rounded-full bg-success/10 text-success flex items-center justify-center shrink-0 mt-0.5">
                                <Moon size={16} />
                            </div>
                            <div>
                                <div className="text-sm font-medium text-primary">Excellent Sleep Quality</div>
                                <div className="text-xs text-secondary mt-1">
                                    Your sleep quality score of {avgQuality.toFixed(0)}% indicates restful, restorative sleep.
                                </div>
                            </div>
                        </div>
                    )}

                    {avgDuration < 7 && (
                        <div className="flex items-start gap-3 p-3 bg-warning/5 rounded-xl border border-warning/10">
                            <div className="size-8 rounded-full bg-warning/10 text-warning flex items-center justify-center shrink-0 mt-0.5">
                                <Moon size={16} />
                            </div>
                            <div>
                                <div className="text-sm font-medium text-primary">Consider More Sleep</div>
                                <div className="text-xs text-secondary mt-1">
                                    You're averaging {avgDuration.toFixed(1)} hours. Try to aim for 7-9 hours for optimal health.
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
