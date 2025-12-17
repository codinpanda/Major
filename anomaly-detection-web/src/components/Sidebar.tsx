import { Activity, Shield, History, Settings, Home, LogOut, Moon, Calendar } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import clsx from 'clsx';

export function Sidebar() {
    const navItems = [
        { to: "/", icon: Home, label: "Home" },
        { to: "/vitals", icon: Activity, label: "Vitals" },
        { to: "/guidance", icon: Shield, label: "Guidance" },
        { to: "/history", icon: History, label: "History" },
        { to: "/calendar", icon: Calendar, label: "Calendar" },
        { to: "/sleep", icon: Moon, label: "Sleep" },
        { to: "/settings", icon: Settings, label: "Settings" },
    ];

    return (
        <aside className="hidden lg:flex flex-col w-64 h-screen bg-surface border-r border-white/5 fixed left-0 top-0 z-50">
            <div className="p-6">
                <div className="flex items-center gap-3">
                    <div className="size-10 rounded-xl bg-accent/10 flex items-center justify-center text-accent">
                        <Activity size={24} />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-primary">HealthGuard</h1>
                        <p className="text-xs text-secondary">Pro Monitor</p>
                    </div>
                </div>
            </div>

            <nav className="flex-1 px-4 space-y-2 mt-4">
                {navItems.map((item) => (
                    <NavLink
                        key={item.to}
                        to={item.to}
                        className={({ isActive }) => clsx(
                            "flex items-center gap-4 px-5 py-4 lg:py-4 rounded-2xl transition-all duration-200",
                            isActive
                                ? "bg-accent/10 text-accent font-medium"
                                : "text-secondary hover:text-primary hover:bg-white/5"
                        )}
                    >
                        <item.icon size={22} className="lg:w-6 lg:h-6" />
                        <span className="lg:text-base">{item.label}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="p-4 border-t border-white/5">
                <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-danger hover:bg-danger/5 transition-colors">
                    <LogOut size={20} />
                    <span>Sign Out</span>
                </button>
            </div>
        </aside>
    );
}
