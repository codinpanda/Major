interface HealthDataPoint {
    timestamp: number;
    hr: number;
    stress: number;
    spo2: number;
    sleep?: number;
}

interface Insight {
    id: string;
    type: 'pattern' | 'correlation' | 'achievement' | 'recommendation';
    severity: 'info' | 'success' | 'warning';
    title: string;
    description: string;
    icon: string;
    timestamp: number;
}

export class InsightsEngine {
    private history: HealthDataPoint[] = [];
    private insights: Insight[] = [];

    addDataPoint(data: HealthDataPoint) {
        this.history.push(data);

        // Keep last 7 days (assuming 1 data point per hour)
        if (this.history.length > 168) {
            this.history.shift();
        }

        // Generate insights periodically
        if (this.history.length % 24 === 0) {
            this.generateInsights();
        }
    }

    private generateInsights() {
        this.insights = [];

        // Pattern: Time-based stress
        this.detectStressPatterns();

        // Pattern: Heart rate patterns
        this.detectHeartRatePatterns();

        // Correlation: Sleep vs Stress
        this.detectSleepStressCorrelation();

        // Achievement: Consistency
        this.detectAchievements();

        // Recommendation: Based on trends
        this.generateRecommendations();

        // Weekly summary
        this.generateWeeklySummary();

        // Hydration reminder
        this.generateHydrationReminder();
    }

    private detectStressPatterns() {
        if (this.history.length < 24) return;

        // Group by hour of day
        const hourlyStress: { [hour: number]: number[] } = {};

        this.history.forEach((point) => {
            const hour = new Date(point.timestamp).getHours();
            if (!hourlyStress[hour]) hourlyStress[hour] = [];
            hourlyStress[hour].push(point.stress);
        });

        // Find peak stress hour
        let peakHour = 0;
        let peakAvg = 0;

        Object.entries(hourlyStress).forEach(([hour, values]) => {
            const avg = values.reduce((a, b) => a + b, 0) / values.length;
            if (avg > peakAvg) {
                peakAvg = avg;
                peakHour = parseInt(hour);
            }
        });

        if (peakAvg > 60) {
            this.insights.push({
                id: 'stress-pattern-1',
                type: 'pattern',
                severity: 'warning',
                title: 'Stress Pattern Detected',
                description: `Your stress tends to peak around ${this.formatHour(peakHour)}. Consider scheduling breaks during this time.`,
                icon: 'clock',
                timestamp: Date.now(),
            });
        }

        // Detect overall stress trend
        const recent7Days = this.history.slice(-168);
        if (recent7Days.length >= 168) {
            const avgStress = recent7Days.reduce((sum, p) => sum + p.stress, 0) / recent7Days.length;

            if (avgStress < 40) {
                this.insights.push({
                    id: 'stress-pattern-2',
                    type: 'achievement',
                    severity: 'success',
                    title: 'Great Stress Management!',
                    description: `Your average stress level this week was ${avgStress.toFixed(0)}%. You're doing an excellent job!`,
                    icon: 'trophy',
                    timestamp: Date.now(),
                });
            }
        }
    }

    private detectHeartRatePatterns() {
        if (this.history.length < 48) return;

        const recent = this.history.slice(-48);
        const avgHR = recent.reduce((sum, p) => sum + p.hr, 0) / recent.length;

        if (avgHR >= 60 && avgHR <= 80) {
            this.insights.push({
                id: 'hr-pattern-1',
                type: 'achievement',
                severity: 'success',
                title: 'Optimal Resting Heart Rate',
                description: `Your average heart rate of ${avgHR.toFixed(0)} BPM is in the optimal range. Keep it up!`,
                icon: 'trending',
                timestamp: Date.now(),
            });
        }
    }

    private detectSleepStressCorrelation() {
        if (this.history.length < 48) return;

        const recentData = this.history.slice(-48);
        let goodSleepLowStress = 0;

        recentData.forEach((point) => {
            if (point.sleep) {
                if (point.sleep >= 7 && point.stress < 50) goodSleepLowStress++;
            }
        });

        if (goodSleepLowStress > 5) {
            this.insights.push({
                id: 'correlation-1',
                type: 'correlation',
                severity: 'success',
                title: 'Sleep Helps Your Stress',
                description: 'You tend to have lower stress on days with 7+ hours of sleep. Keep it up!',
                icon: 'moon',
                timestamp: Date.now(),
            });
        }
    }

    private detectAchievements() {
        if (this.history.length < 168) return;

        const last7Days = this.history.slice(-168);
        const avgHR = last7Days.reduce((sum, p) => sum + p.hr, 0) / last7Days.length;
        const avgStress = last7Days.reduce((sum, p) => sum + p.stress, 0) / last7Days.length;

        if (avgHR >= 60 && avgHR <= 80 && avgStress < 50) {
            this.insights.push({
                id: 'achievement-1',
                type: 'achievement',
                severity: 'success',
                title: '7-Day Healthy Streak!',
                description: 'Your heart rate and stress have been consistently healthy this week.',
                icon: 'trophy',
                timestamp: Date.now(),
            });
        }
    }

    private generateRecommendations() {
        if (this.history.length < 24) return;

        const recent = this.history.slice(-24);
        const avgStress = recent.reduce((sum, p) => sum + p.stress, 0) / recent.length;

        if (avgStress > 65) {
            this.insights.push({
                id: 'recommendation-1',
                type: 'recommendation',
                severity: 'info',
                title: 'Try a Breathing Exercise',
                description: 'Your stress has been elevated today. A 5-minute breathing exercise might help you feel calmer.',
                icon: 'wind',
                timestamp: Date.now(),
            });
        }

        const avgHR = recent.reduce((sum, p) => sum + p.hr, 0) / recent.length;
        if (avgHR > 85) {
            this.insights.push({
                id: 'recommendation-2',
                type: 'recommendation',
                severity: 'info',
                title: 'Take It Easy',
                description: 'Your heart rate has been higher than usual. Consider taking a short break to relax.',
                icon: 'activity',
                timestamp: Date.now(),
            });
        }
    }

    private generateWeeklySummary() {
        if (this.history.length < 336) return;

        const last7Days = this.history.slice(-168);
        const previous7Days = this.history.slice(-336, -168);

        const avgStress = last7Days.reduce((sum, p) => sum + p.stress, 0) / last7Days.length;
        const prevAvgStress = previous7Days.reduce((sum, p) => sum + p.stress, 0) / previous7Days.length;
        const stressChange = ((avgStress - prevAvgStress) / prevAvgStress) * 100;

        if (Math.abs(stressChange) > 10) {
            const improved = stressChange < 0;
            this.insights.push({
                id: 'weekly-summary-1',
                type: 'pattern',
                severity: improved ? 'success' : 'info',
                title: 'Weekly Stress Trend',
                description: improved
                    ? `Great news! Your stress decreased by ${Math.abs(stressChange).toFixed(0)}% this week.`
                    : `Your stress increased by ${stressChange.toFixed(0)}% this week. Consider what might be causing this.`,
                icon: 'trending',
                timestamp: Date.now(),
            });
        }
    }

    private generateHydrationReminder() {
        const currentHour = new Date().getHours();
        const recent = this.history.slice(-6);

        if (recent.length < 6) return;

        const avgHR = recent.reduce((sum, p) => sum + p.hr, 0) / recent.length;

        if (currentHour >= 10 && currentHour <= 18 && avgHR > 75) {
            this.insights.push({
                id: 'hydration-1',
                type: 'recommendation',
                severity: 'info',
                title: 'Stay Hydrated',
                description: 'Your heart rate has been slightly elevated. Make sure you\'re drinking enough water throughout the day.',
                icon: 'activity',
                timestamp: Date.now(),
            });
        }
    }

    private formatHour(hour: number): string {
        const period = hour >= 12 ? 'PM' : 'AM';
        const displayHour = hour % 12 || 12;
        return `${displayHour}:00 ${period}`;
    }

    getInsights(): Insight[] {
        return this.insights.sort((a, b) => b.timestamp - a.timestamp);
    }

    getInsightsByType(type: Insight['type']): Insight[] {
        return this.insights.filter((i) => i.type === type);
    }

    clearInsights() {
        this.insights = [];
    }
}

// Singleton instance
export const insightsEngine = new InsightsEngine();
