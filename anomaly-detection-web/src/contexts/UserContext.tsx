import React, { createContext, useContext, useState, useEffect } from 'react';

export interface LogEntry {
    id: string;
    type: 'exercise' | 'meal';
    title: string;
    details: string;
    timestamp: string;
}

interface User {
    name: string;
    email: string;
    avatar?: string;
    logs: LogEntry[];
    settings: {
        notifications: {
            alerts: boolean;
            summary: boolean;
            reminders: boolean;
        };
        thresholds: {
            maxHeartRate: number;
            stressAlert: number;
        };
    };
}

interface UserContextType {
    user: User;
    updateUser: (updates: Partial<User> | ((prev: User) => Partial<User>)) => void;
    addLog: (entry: Omit<LogEntry, 'id' | 'timestamp'>) => void;
}

const defaultUser: User = {
    name: 'Priyanshu',
    email: 'priyanshu@example.com',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Priyanshu',
    logs: [],
    settings: {
        notifications: {
            alerts: true,
            summary: true,
            reminders: false
        },
        thresholds: {
            maxHeartRate: 120,
            stressAlert: 70
        }
    }
};

const UserContext = createContext<UserContextType | undefined>(undefined);

export function UserProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User>(() => {
        const saved = localStorage.getItem('user_profile');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                // Deep merge/ensure settings exist
                return {
                    ...defaultUser,
                    ...parsed,
                    logs: parsed.logs || [],
                    settings: {
                        ...defaultUser.settings,
                        ...(parsed.settings || {}),
                        notifications: {
                            ...defaultUser.settings.notifications,
                            ...(parsed.settings?.notifications || {})
                        },
                        thresholds: {
                            ...defaultUser.settings.thresholds,
                            ...(parsed.settings?.thresholds || {})
                        }
                    }
                };
            } catch (e) {
                console.error('Failed to parse user profile', e);
                return defaultUser;
            }
        }
        return defaultUser;
    });

    useEffect(() => {
        localStorage.setItem('user_profile', JSON.stringify(user));
    }, [user]);

    const updateUser = (updates: Partial<User> | ((prev: User) => Partial<User>)) => {
        setUser(prev => {
            const newValues = typeof updates === 'function' ? updates(prev) : updates;
            // Deep merge for settings to avoid overwriting nested objects
            if (newValues.settings) {
                return {
                    ...prev,
                    ...newValues,
                    settings: {
                        ...prev.settings,
                        ...newValues.settings,
                        notifications: {
                            ...prev.settings.notifications,
                            ...(newValues.settings.notifications || {})
                        },
                        thresholds: {
                            ...prev.settings.thresholds,
                            ...(newValues.settings.thresholds || {})
                        }
                    }
                };
            }
            return { ...prev, ...newValues };
        });
    };

    const addLog = (entry: Omit<LogEntry, 'id' | 'timestamp'>) => {
        const newLog: LogEntry = {
            ...entry,
            id: Date.now().toString(),
            timestamp: new Date().toISOString()
        };
        setUser(prev => ({
            ...prev,
            logs: [newLog, ...(prev.logs || [])]
        }));
    };

    return (
        <UserContext.Provider value={{ user, updateUser, addLog }}>
            {children}
        </UserContext.Provider>
    );
}

export function useUser() {
    const context = useContext(UserContext);
    if (context === undefined) {
        throw new Error('useUser must be used within a UserProvider');
    }
    return context;
}
