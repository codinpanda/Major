import { User, Bell, Activity, Shield, LogOut, ChevronRight, Share2, Watch } from 'lucide-react';
import { Toggle } from '../components/Toggle';
import { useUser } from '../contexts/UserContext';
import { dataGenerator } from '../simulation/DataGenerator';
import { useState } from 'react';

export function Settings() {
    const { user, updateUser } = useUser();
    // Use settings from context or fallbacks
    const notifications = user.settings?.notifications || {
        alerts: true,
        summary: true,
        reminders: false
    };
    const thresholds = user.settings?.thresholds || {
        maxHeartRate: 120,
        stressAlert: 70
    };

    const [isWatchConnected, setWatchConnected] = useState(dataGenerator.mode === 'EXTERNAL_DEVICE');

    const toggleWatchConnection = (enabled: boolean) => {
        setWatchConnected(enabled);
        dataGenerator.setMode(enabled ? 'EXTERNAL_DEVICE' : 'NORMAL');
    };

    const updateNotifications = (key: keyof typeof notifications, value: boolean) => {
        updateUser(prev => ({
            settings: {
                ...prev.settings,
                notifications: {
                    ...(prev.settings?.notifications || notifications),
                    [key]: value
                },
                thresholds: prev.settings?.thresholds || thresholds
            }
        }));
    };

    const updateThresholds = (key: keyof typeof thresholds, value: number) => {
        updateUser(prev => ({
            settings: {
                ...prev.settings,
                notifications: prev.settings?.notifications || notifications,
                thresholds: {
                    ...(prev.settings?.thresholds || thresholds),
                    [key]: value
                }
            }
        }));
    };

    return (
        <div className="space-y-6 w-full max-w-7xl mx-auto">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold text-primary">Settings</h2>
                    <p className="text-secondary mt-1">Manage preferences & device</p>
                </div>
                <button className="flex items-center gap-2 px-4 py-2 bg-surface hover:bg-surfaceHover border border-white/10 rounded-xl text-primary font-medium transition-colors">
                    <Share2 size={18} /> Share Profile
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column (Profile & Goals) */}
                <div className="space-y-6">
                    {/* Hero Profile Card */}
                    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-accent/20 to-surface border border-white/5 p-6 lg:p-8">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-accent/20 blur-[50px] rounded-full pointer-events-none" />

                        <div className="relative z-10 flex flex-col items-center text-center">
                            <div className="relative mb-4">
                                <div className="size-24 lg:size-28 rounded-full bg-gradient-to-br from-accent to-purple-500 p-1 shadow-xl shadow-accent/20">
                                    <div className="w-full h-full rounded-full bg-surface flex items-center justify-center text-3xl font-bold text-white overflow-hidden">
                                        <img src={user.avatar} alt="Avatar" className="w-full h-full" />
                                    </div>
                                </div>
                                <button className="absolute bottom-0 right-0 p-2 bg-primary text-surface rounded-full shadow-lg hover:scale-110 transition-transform">
                                    <User size={14} />
                                </button>
                            </div>

                            <h3 className="text-2xl font-bold text-primary">{user.name}</h3>
                            <p className="text-secondary mb-6">Premium Member</p>

                            <div className="grid grid-cols-3 gap-4 w-full pt-6 border-t border-white/10">
                                <div>
                                    <div className="text-2xl font-bold text-primary">28</div>
                                    <div className="text-xs text-secondary uppercase tracking-wide mt-1">Age</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-bold text-primary">70kg</div>
                                    <div className="text-xs text-secondary uppercase tracking-wide mt-1">Weight</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-bold text-primary">178cm</div>
                                    <div className="text-xs text-secondary uppercase tracking-wide mt-1">Height</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Sign Out */}
                    <button className="w-full card-base p-4 bg-surface/50 hover:bg-danger/10 hover:border-danger/20 transition-all flex items-center justify-center gap-2 text-danger group">
                        <LogOut size={20} className="group-hover:-translate-x-1 transition-transform" />
                        <span className="font-medium">Sign Out</span>
                    </button>
                </div>

                {/* Right Column (Settings Grid) */}
                <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Personal Info */}
                    <div className="card-base p-6 bg-surface md:col-span-2">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                                <User size={20} />
                            </div>
                            <h3 className="font-bold text-primary text-lg">Personal Details</h3>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-secondary">Full Name</label>
                                <input
                                    type="text"
                                    value={user.name}
                                    onChange={(e) => updateUser({ name: e.target.value })}
                                    className="w-full bg-background border border-white/5 rounded-xl px-4 py-3 text-primary focus:outline-none focus:border-accent/50 focus:bg-accent/5 transition-all placeholder:text-white/20"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-secondary">Email</label>
                                <input
                                    type="email"
                                    value={user.email}
                                    onChange={(e) => updateUser({ email: e.target.value })}
                                    className="w-full bg-background border border-white/5 rounded-xl px-4 py-3 text-primary focus:outline-none focus:border-accent/50 focus:bg-accent/5 transition-all placeholder:text-white/20"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Device Connection */}
                    <div className="card-base p-6 bg-surface md:col-span-2">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-blue-500/10 text-blue-500 flex items-center justify-center">
                                <Watch size={20} />
                            </div>
                            <h3 className="font-bold text-primary text-lg">Device Connection</h3>
                        </div>
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="font-medium text-primary">Samsung Watch Integration</div>
                                <div className="text-xs text-secondary mt-0.5">Stream real-time data from local bridge</div>
                            </div>
                            <Toggle checked={isWatchConnected} onChange={toggleWatchConnection} />
                        </div>
                    </div>

                    {/* Notifications */}
                    <div className="card-base p-6 bg-surface">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-warning/10 text-warning flex items-center justify-center">
                                <Bell size={20} />
                            </div>
                            <h3 className="font-bold text-primary text-lg">Notifications</h3>
                        </div>
                        <div className="space-y-5">
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="font-medium text-primary">Health Alerts</div>
                                    <div className="text-xs text-secondary mt-0.5">Unusual patterns detected</div>
                                </div>
                                <Toggle checked={notifications.alerts} onChange={(c) => updateNotifications('alerts', c)} />
                            </div>
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="font-medium text-primary">Daily Summary</div>
                                    <div className="text-xs text-secondary mt-0.5">Morning health report</div>
                                </div>
                                <Toggle checked={notifications.summary} onChange={(c) => updateNotifications('summary', c)} />
                            </div>
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="font-medium text-primary">Reminders</div>
                                    <div className="text-xs text-secondary mt-0.5">Breathing & movement</div>
                                </div>
                                <Toggle checked={notifications.reminders} onChange={(c) => updateNotifications('reminders', c)} />
                            </div>
                        </div>
                    </div>

                    {/* Personalization (Flagship Phase 2) */}
                    <div className="card-base p-6 bg-surface">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-purple-500/10 text-purple-500 flex items-center justify-center">
                                <Activity size={20} />
                            </div>
                            <h3 className="font-bold text-primary text-lg">Health Model</h3>
                        </div>
                        <div className="space-y-5">
                            {/* Resting HR Input */}
                            <div>
                                <label className="text-sm font-medium text-secondary mb-2 block">Resting Heart Rate (BPM)</label>
                                <div className="flex items-center gap-4">
                                    <input
                                        type="number"
                                        className="bg-background border border-white/5 rounded-xl px-4 py-2 text-primary w-24 focus:outline-none focus:border-accent"
                                        placeholder="60"
                                        defaultValue={60}
                                    />
                                    <span className="text-xs text-secondary">Used as baseline for anomaly detection.</span>
                                </div>
                            </div>

                            {/* Risk Sensitivity Slider */}
                            <div>
                                <div className="flex justify-between items-center mb-2">
                                    <label className="text-sm font-medium text-secondary">Risk Sensitivity</label>
                                    <span className="text-sm font-bold text-accent">Medium</span>
                                </div>
                                <input
                                    type="range"
                                    min="1"
                                    max="3"
                                    step="1"
                                    defaultValue="2"
                                    className="w-full accent-accent h-2 bg-background rounded-lg appearance-none cursor-pointer"
                                />
                                <div className="flex justify-between text-[10px] text-secondary mt-1 uppercase font-bold tracking-wider">
                                    <span>Low</span>
                                    <span>Medium</span>
                                    <span>High</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Thresholds */}
                    <div className="card-base p-6 bg-surface">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-danger/10 text-danger flex items-center justify-center">
                                <Shield size={20} />
                            </div>
                            <h3 className="font-bold text-primary text-lg">Safety Thresholds</h3>
                        </div>
                        <div className="space-y-5">
                            <div>
                                <div className="flex justify-between items-center mb-2">
                                    <label className="text-sm font-medium text-secondary">Max Heart Rate</label>
                                    <span className="text-sm font-bold text-primary">{thresholds.maxHeartRate} BPM</span>
                                </div>
                                <input
                                    type="range"
                                    min="100"
                                    max="200"
                                    value={thresholds.maxHeartRate}
                                    onChange={(e) => updateThresholds('maxHeartRate', parseInt(e.target.value))}
                                    className="w-full accent-danger h-2 bg-background rounded-lg appearance-none cursor-pointer"
                                />
                            </div>
                            <div>
                                <div className="flex justify-between items-center mb-2">
                                    <label className="text-sm font-medium text-secondary">Stress Alert</label>
                                    <span className="text-sm font-bold text-primary">{thresholds.stressAlert}%</span>
                                </div>
                                <input
                                    type="range"
                                    min="50"
                                    max="100"
                                    value={thresholds.stressAlert}
                                    onChange={(e) => updateThresholds('stressAlert', parseInt(e.target.value))}
                                    className="w-full accent-warning h-2 bg-background rounded-lg appearance-none cursor-pointer"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Privacy Zone (Flagship Phase 2) */}
                    <div className="card-base p-6 bg-surface md:col-span-2 border border-danger/20">
                        <div className="flex items-center gap-3 mb-4">
                            <div className="size-10 rounded-xl bg-danger/10 text-danger flex items-center justify-center">
                                <Shield size={20} />
                            </div>
                            <div>
                                <h3 className="font-bold text-primary text-lg">Privacy & Data</h3>
                                <p className="text-xs text-secondary">Manage your local data storage.</p>
                            </div>
                        </div>
                        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                            <p className="text-sm text-secondary max-w-md">
                                All health data is currently stored locally on this device. Deleting it is permanent and cannot be undone.
                            </p>
                            <button
                                className="px-4 py-2 bg-danger/10 hover:bg-danger text-danger hover:text-white border border-danger/20 rounded-xl font-medium transition-all text-sm whitespace-nowrap"
                                onClick={() => {
                                    if (confirm("Are you sure? This will wipe all history.")) {
                                        alert("Data cleared.");
                                    }
                                }}
                            >
                                Delete My Data
                            </button>
                        </div>
                    </div>

                    {/* Health Goals */}
                    <div className="card-base p-6 bg-surface md:col-span-2">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="size-10 rounded-xl bg-success/10 text-success flex items-center justify-center">
                                <Activity size={20} />
                            </div>
                            <h3 className="font-bold text-primary text-lg">Daily Goals</h3>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="p-4 rounded-2xl bg-background border border-white/5 flex items-center justify-between group cursor-pointer hover:border-white/10 transition-colors">
                                <div>
                                    <div className="text-xs text-secondary uppercase tracking-wide">Steps</div>
                                    <div className="text-xl font-bold text-primary mt-1">10,000</div>
                                </div>
                                <ChevronRight className="text-white/20 group-hover:text-white/40 transition-colors" />
                            </div>
                            <div className="p-4 rounded-2xl bg-background border border-white/5 flex items-center justify-between group cursor-pointer hover:border-white/10 transition-colors">
                                <div>
                                    <div className="text-xs text-secondary uppercase tracking-wide">Sleep</div>
                                    <div className="text-xl font-bold text-primary mt-1">8 Hours</div>
                                </div>
                                <ChevronRight className="text-white/20 group-hover:text-white/40 transition-colors" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div >
    );
}
