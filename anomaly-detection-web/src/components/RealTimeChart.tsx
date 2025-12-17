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
import type { ChartData, ChartOptions, TooltipItem } from 'chart.js';
import { Line } from 'react-chartjs-2';

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
import annotationPlugin from 'chartjs-plugin-annotation';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(annotationPlugin, zoomPlugin);

import { ExportMenu } from './ExportMenu';

interface RealTimeChartProps {
    data: number[];
    labels: string[];
    label: string;
    color: string;
    min?: number;
    max?: number;
    normalMin?: number;
    normalMax?: number;
    unit?: string;
    id?: string;
}

export function RealTimeChart({ data, labels, label, color, min, max, normalMin, normalMax, unit = '', id }: RealTimeChartProps) {
    // Detect mobile device for responsive options
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
    // Generate a default ID if none provided, using a simple random string or label-based ID
    const chartId = id || `chart-${label.replace(/\s+/g, '-').toLowerCase()}-${Math.random().toString(36).substr(2, 9)}`;

    const options: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        interaction: {
            mode: 'index',
            intersect: false,
        },
        scales: {
            x: {
                grid: {
                    color: '#FFFFFF08', // Very subtle
                    borderDash: [5, 5],
                    tickLength: 0
                } as any,
                ticks: {
                    color: '#9ca3af',
                    font: {
                        size: isMobile ? 9 : 10,
                        family: 'Inter',
                    },
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: isMobile ? 3 : 6 // Show fewer labels (e.g. 60s, 40s, 20s, Now)
                }
            },
            y: {
                min,
                max,
                grid: {
                    color: '#FFFFFF08',
                    borderDash: [5, 5],
                    lineWidth: 1,
                    tickLength: 0
                } as any,
                border: { display: false },
                ticks: {
                    color: '#9ca3af',
                    font: {
                        size: isMobile ? 9 : 11,
                        family: 'Inter',
                        weight: 'bold'
                    },
                    padding: isMobile ? 4 : 8,
                    maxTicksLimit: isMobile ? 5 : 6
                }
            },
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                enabled: true,
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(28, 28, 30, 0.95)',
                titleColor: '#F2F2F7',
                bodyColor: '#98989D',
                borderColor: '#00C7BE',
                borderWidth: 2,
                padding: isMobile ? 10 : 12,
                cornerRadius: 12,
                displayColors: false,
                titleFont: {
                    size: isMobile ? 12 : 13,
                    weight: 'bold',
                    family: 'Inter'
                },
                bodyFont: {
                    size: isMobile ? 11 : 12,
                    family: 'Inter'
                },
                callbacks: {
                    title: (tooltipItems: TooltipItem<'line'>[]) => {
                        // Use the label directly since it's now friendly ("15s ago")
                        return tooltipItems[0].label;
                    },
                    afterLabel: (_: TooltipItem<'line'>) => {
                        if (normalMin && normalMax) {
                            return `Normal: ${normalMin}-${normalMax}${unit}`;
                        }
                        return '';
                    }
                }
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'x',
                    modifierKey: isMobile ? undefined : 'ctrl'
                },
                zoom: {
                    wheel: { enabled: !isMobile },
                    pinch: { enabled: true },
                    mode: 'x',
                }
            },
            annotation: {
                annotations: (normalMin !== undefined && normalMax !== undefined) ? {
                    normalRange: {
                        type: 'box',
                        yMin: normalMin,
                        yMax: normalMax,
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        borderWidth: 0,
                        label: {
                            content: 'Normal',
                            display: !isMobile,
                            position: 'start',
                            color: 'rgba(16, 185, 129, 0.5)',
                            font: { size: 9 },
                            yAdjust: -10
                        }
                    }
                } : {}
            }
        },
        elements: {
            point: {
                radius: 0,
                hitRadius: 20,
                hoverRadius: 6,
                hoverBorderWidth: 2,
                hoverBackgroundColor: color,
                hoverBorderColor: '#ffffff'
            },
            line: {
                tension: 0.4, // Organic curve
                borderWidth: isMobile ? 2 : 3,
                borderCapStyle: 'round',
                borderJoinStyle: 'round',
            }
        }
    };

    const chartData: ChartData<'line'> = {
        labels,
        datasets: [
            {
                label,
                data,
                borderColor: color,
                backgroundColor: (context) => {
                    const ctx = context.chart.ctx;
                    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
                    gradient.addColorStop(0, color + '40'); // 25% opacity
                    gradient.addColorStop(1, color + '00'); // Fade out
                    return gradient;
                },
                fill: true,
            },
        ],
    };

    return (
        <div className="w-full h-full relative group">
            <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                <ExportMenu
                    chartId={chartId}
                    chartTitle={label}
                    data={data.map((d, i) => ({ time: labels[i], value: d }))}
                    headers={['Time', label]}
                />
            </div>
            <Line id={chartId} options={options} data={chartData} />
        </div>
    );
}

