import { useState } from 'react';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface HealthDay {
    date: string; // YYYY-MM-DD
    status: 'good' | 'fair' | 'poor';
    avgHR: number;
    avgStress: number;
    sleep?: number;
}

export function CalendarView() {
    const [selectedDate, setSelectedDate] = useState<Date>(new Date());

    // Mock health data for the month
    const healthData: HealthDay[] = generateMockHealthData();

    const getHealthStatus = (date: Date): HealthDay | undefined => {
        const dateStr = date.toISOString().split('T')[0];
        return healthData.find(d => d.date === dateStr);
    };

    const getTileClassName = ({ date }: { date: Date }) => {
        const health = getHealthStatus(date);
        if (!health) return '';

        if (health.status === 'good') return 'health-good';
        if (health.status === 'fair') return 'health-fair';
        return 'health-poor';
    };

    const selectedDayData = getHealthStatus(selectedDate);

    return (
        <div className="space-y-4 w-full">
            {/* Header */}
            <div className="card-base p-6 bg-surface">
                <h2 className="text-xl font-bold text-primary">Health Calendar</h2>
                <p className="text-secondary text-sm mt-2">Track your health journey day by day</p>
            </div>

            {/* Legend */}
            <div className="card-base p-4 bg-surface">
                <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex items-center gap-2">
                        <div className="size-4 rounded-full bg-success"></div>
                        <span className="text-xs text-secondary">Good Day</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="size-4 rounded-full bg-warning"></div>
                        <span className="text-xs text-secondary">Fair Day</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="size-4 rounded-full bg-danger"></div>
                        <span className="text-xs text-secondary">Needs Attention</span>
                    </div>
                </div>
            </div>

            {/* Calendar */}
            <div className="card-base p-5 bg-surface calendar-container">
                <Calendar
                    onChange={(value) => setSelectedDate(value as Date)}
                    value={selectedDate}
                    tileClassName={getTileClassName}
                    className="health-calendar"
                />
            </div>

            {/* Selected Day Details */}
            {selectedDayData && (
                <div className="card-base p-5 bg-surface animate-fade-in">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="font-bold text-primary">
                            {selectedDate.toLocaleDateString('en-US', {
                                weekday: 'long',
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                            })}
                        </h3>
                        <div className={`px-3 py-1 rounded-full text-xs font-medium ${selectedDayData.status === 'good' ? 'bg-success/10 text-success' :
                                selectedDayData.status === 'fair' ? 'bg-warning/10 text-warning' :
                                    'bg-danger/10 text-danger'
                            }`}>
                            {selectedDayData.status === 'good' ? 'Good Day' :
                                selectedDayData.status === 'fair' ? 'Fair Day' : 'Needs Attention'}
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                        <div className="text-center p-3 bg-background rounded-xl">
                            <div className="text-xs text-secondary mb-1">Heart Rate</div>
                            <div className="text-2xl font-bold text-primary">{selectedDayData.avgHR}</div>
                            <div className="text-xs text-secondary">BPM</div>
                        </div>
                        <div className="text-center p-3 bg-background rounded-xl">
                            <div className="text-xs text-secondary mb-1">Stress</div>
                            <div className="text-2xl font-bold text-primary">{selectedDayData.avgStress}</div>
                            <div className="text-xs text-secondary">%</div>
                        </div>
                        <div className="text-center p-3 bg-background rounded-xl">
                            <div className="text-xs text-secondary mb-1">Sleep</div>
                            <div className="text-2xl font-bold text-primary">
                                {selectedDayData.sleep?.toFixed(1) || '-'}
                            </div>
                            <div className="text-xs text-secondary">hrs</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Monthly Summary */}
            <div className="card-base p-5 bg-surface">
                <h3 className="font-bold text-primary mb-4">This Month</h3>
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-secondary">Good Days</span>
                        <div className="flex items-center gap-2">
                            <span className="text-lg font-bold text-success">
                                {healthData.filter(d => d.status === 'good').length}
                            </span>
                            <TrendingUp size={16} className="text-success" />
                        </div>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-secondary">Fair Days</span>
                        <div className="flex items-center gap-2">
                            <span className="text-lg font-bold text-warning">
                                {healthData.filter(d => d.status === 'fair').length}
                            </span>
                            <Minus size={16} className="text-warning" />
                        </div>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-secondary">Days Needing Attention</span>
                        <div className="flex items-center gap-2">
                            <span className="text-lg font-bold text-danger">
                                {healthData.filter(d => d.status === 'poor').length}
                            </span>
                            <TrendingDown size={16} className="text-danger" />
                        </div>
                    </div>
                </div>
            </div>

            <style>{`
                .calendar-container {
                    --react-calendar-bg: transparent;
                }
                
                .health-calendar {
                    width: 100%;
                    border: none;
                    background: transparent;
                    font-family: inherit;
                }
                
                .health-calendar button {
                    color: var(--color-primary, #F2F2F7);
                    border-radius: 8px;
                    margin: 2px;
                }
                
                .health-calendar .react-calendar__tile--active {
                    background: var(--color-accent, #00C7BE) !important;
                    color: var(--color-surface, #2C2C2E) !important;
                }
                
                .health-calendar .react-calendar__tile--now {
                    background: rgba(0, 199, 190, 0.1);
                }
                
                .health-calendar .health-good {
                    background: rgba(48, 209, 88, 0.2) !important;
                    border: 2px solid #30D158;
                }
                
                .health-calendar .health-fair {
                    background: rgba(255, 214, 10, 0.2) !important;
                    border: 2px solid #FFD60A;
                }
                
                .health-calendar .health-poor {
                    background: rgba(255, 69, 58, 0.2) !important;
                    border: 2px solid #FF453A;
                }
                
                .health-calendar .react-calendar__month-view__weekdays {
                    color: var(--color-secondary, #98989D);
                    font-size: 0.75rem;
                    text-transform: uppercase;
                }
                
                .health-calendar .react-calendar__navigation button {
                    color: var(--color-primary, #F2F2F7);
                    font-size: 1rem;
                    font-weight: 600;
                }
                
                .health-calendar .react-calendar__navigation button:hover {
                    background: rgba(0, 199, 190, 0.1);
                }
            `}</style>
        </div>
    );
}

// Generate mock health data for the current month
function generateMockHealthData(): HealthDay[] {
    const data: HealthDay[] = [];
    const today = new Date();
    const daysInMonth = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();

    for (let day = 1; day <= Math.min(daysInMonth, today.getDate()); day++) {
        const date = new Date(today.getFullYear(), today.getMonth(), day);
        const dateStr = date.toISOString().split('T')[0];

        const avgHR = 60 + Math.floor(Math.random() * 30);
        const avgStress = 30 + Math.floor(Math.random() * 40);
        const sleep = 6 + Math.random() * 3;

        let status: 'good' | 'fair' | 'poor' = 'good';
        if (avgHR > 90 || avgStress > 60) status = 'fair';
        if (avgHR > 100 || avgStress > 75) status = 'poor';

        data.push({
            date: dateStr,
            status,
            avgHR,
            avgStress,
            sleep,
        });
    }

    return data;
}
