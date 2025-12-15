interface Achievement {
    id: string;
    title: string;
    description: string;
    icon: string;
    unlocked: boolean;
    unlockedAt?: number;
    progress: number; // 0-100
    requirement: string;
}

export class AchievementSystem {
    private achievements: Achievement[] = [
        {
            id: 'streak-3',
            title: '3-Day Streak',
            description: 'Maintained healthy vitals for 3 days',
            icon: '🔥',
            unlocked: false,
            progress: 0,
            requirement: '3 consecutive days with normal heart rate and low stress',
        },
        {
            id: 'streak-7',
            title: 'Week Warrior',
            description: 'Maintained healthy vitals for 7 days',
            icon: '⭐',
            unlocked: false,
            progress: 0,
            requirement: '7 consecutive days with normal heart rate and low stress',
        },
        {
            id: 'sleep-master',
            title: 'Sleep Master',
            description: 'Got 7+ hours of sleep for 5 nights',
            icon: '😴',
            unlocked: false,
            progress: 0,
            requirement: '5 nights with 7+ hours of sleep',
        },
        {
            id: 'stress-free',
            title: 'Stress-Free Week',
            description: 'Kept stress below 50% for a week',
            icon: '🧘',
            unlocked: false,
            progress: 0,
            requirement: '7 days with stress level below 50%',
        },
        {
            id: 'heart-health',
            title: 'Heart Health Champion',
            description: 'Maintained optimal heart rate for 14 days',
            icon: '❤️',
            unlocked: false,
            progress: 0,
            requirement: '14 days with heart rate 60-100 BPM',
        },
        {
            id: 'early-bird',
            title: 'Early Bird',
            description: 'Woke up before 7 AM for 7 days',
            icon: '🌅',
            unlocked: false,
            progress: 0,
            requirement: '7 days waking up before 7:00 AM',
        },
        {
            id: 'breathing-pro',
            title: 'Breathing Pro',
            description: 'Completed 10 breathing exercises',
            icon: '💨',
            unlocked: false,
            progress: 0,
            requirement: '10 breathing exercise sessions',
        },
        {
            id: 'first-month',
            title: 'First Month',
            description: 'Used the app for 30 days',
            icon: '📅',
            unlocked: false,
            progress: 0,
            requirement: '30 days of activity tracking',
        },
    ];

    private streakData = {
        currentStreak: 0,
        lastCheckDate: '',
        healthyDays: 0,
        sleepDays: 0,
        stressDays: 0,
        heartDays: 0,
        breathingSessions: 0,
        totalDays: 0,
    };

    updateProgress(data: {
        hr?: number;
        stress?: number;
        sleep?: number;
        timestamp: number;
    }) {
        const today = new Date(data.timestamp).toDateString();

        // Check if it's a new day
        if (this.streakData.lastCheckDate !== today) {
            this.streakData.lastCheckDate = today;
            this.streakData.totalDays++;

            // Check health metrics
            const isHealthy = data.hr && data.hr >= 60 && data.hr <= 100 &&
                data.stress && data.stress < 50;

            if (isHealthy) {
                this.streakData.currentStreak++;
                this.streakData.healthyDays++;
            } else {
                this.streakData.currentStreak = 0;
            }

            // Sleep tracking
            if (data.sleep && data.sleep >= 7) {
                this.streakData.sleepDays++;
            }

            // Stress tracking
            if (data.stress && data.stress < 50) {
                this.streakData.stressDays++;
            }

            // Heart rate tracking
            if (data.hr && data.hr >= 60 && data.hr <= 100) {
                this.streakData.heartDays++;
            }

            this.checkAchievements();
        }
    }

    incrementBreathingSessions() {
        this.streakData.breathingSessions++;
        this.checkAchievements();
    }

    private checkAchievements() {
        // 3-Day Streak
        this.updateAchievement('streak-3', this.streakData.currentStreak >= 3,
            (this.streakData.currentStreak / 3) * 100);

        // 7-Day Streak
        this.updateAchievement('streak-7', this.streakData.currentStreak >= 7,
            (this.streakData.currentStreak / 7) * 100);

        // Sleep Master
        this.updateAchievement('sleep-master', this.streakData.sleepDays >= 5,
            (this.streakData.sleepDays / 5) * 100);

        // Stress-Free Week
        this.updateAchievement('stress-free', this.streakData.stressDays >= 7,
            (this.streakData.stressDays / 7) * 100);

        // Heart Health Champion
        this.updateAchievement('heart-health', this.streakData.heartDays >= 14,
            (this.streakData.heartDays / 14) * 100);

        // Breathing Pro
        this.updateAchievement('breathing-pro', this.streakData.breathingSessions >= 10,
            (this.streakData.breathingSessions / 10) * 100);

        // First Month
        this.updateAchievement('first-month', this.streakData.totalDays >= 30,
            (this.streakData.totalDays / 30) * 100);
    }

    private updateAchievement(id: string, unlocked: boolean, progress: number) {
        const achievement = this.achievements.find((a) => a.id === id);
        if (!achievement) return;

        achievement.progress = Math.min(100, progress);

        if (unlocked && !achievement.unlocked) {
            achievement.unlocked = true;
            achievement.unlockedAt = Date.now();
        }
    }

    getAchievements(): Achievement[] {
        return this.achievements;
    }

    getUnlockedAchievements(): Achievement[] {
        return this.achievements.filter((a) => a.unlocked);
    }

    getRecentlyUnlocked(hours: number = 24): Achievement[] {
        const cutoff = Date.now() - hours * 60 * 60 * 1000;
        return this.achievements.filter(
            (a) => a.unlocked && a.unlockedAt && a.unlockedAt > cutoff
        );
    }

    getProgress(): typeof this.streakData {
        return { ...this.streakData };
    }
}

// Singleton instance
export const achievementSystem = new AchievementSystem();
