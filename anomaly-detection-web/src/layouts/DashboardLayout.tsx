import { Activity, Shield, History, Settings, Home } from 'lucide-react';
import { Sun, Moon } from 'lucide-react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import clsx from 'clsx';
import { ChatAssistant } from '../components/ChatAssistant';
import { useTheme } from '../contexts/ThemeContext';
import { Sidebar } from '../components/Sidebar';

function NavItem({ to, icon: Icon, label }: { to: string, icon: React.ElementType, label: string }) {
    const location = useLocation();
    const isActive = location.pathname === to;

    return (
        <Link
            to={to}
            className={clsx(
                "flex flex-col items-center gap-1 p-2 rounded-2xl transition-all duration-300 tap-target no-select",
                "min-w-[60px] sm:min-w-[70px]", // Touch-friendly minimum width
                isActive
                    ? "text-primary bg-white/5"
                    : "text-secondary hover:text-primary hover:bg-white/5"
            )}
        >
            <Icon size={24} strokeWidth={isActive ? 2.5 : 2} className="sm:w-6 sm:h-6" />
            <span className="text-[10px] sm:text-xs font-medium tracking-wide">{label}</span>
        </Link>
    );
}

export function DashboardLayout() {
    const { theme, toggleTheme } = useTheme();

    return (
        <div className="flex flex-col lg:flex-row h-screen bg-background text-primary font-sans overflow-hidden">
            {/* Desktop Sidebar */}
            <Sidebar />

            <div className="flex-1 flex flex-col h-full lg:ml-64 relative transition-all duration-300">
                {/* Header - Responsive */}
                <header className="flex-none h-16 sm:h-20 md:h-24 bg-background/80 backdrop-blur-md flex items-center justify-between px-4 sm:px-6 z-40 border-b border-white/5 lg:px-8">
                    <div className="min-w-0 flex-1">
                        <h1 className="text-lg sm:text-xl md:text-2xl font-bold text-primary truncate">
                            Good Morning, Priyanshu
                        </h1>
                        <div className="flex items-center gap-2 mt-0.5 sm:mt-1">
                            <span className="flex size-2 rounded-full bg-accent animate-pulse"></span>
                            <span className="text-[10px] sm:text-xs text-secondary font-medium">Monitoring Active</span>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 sm:gap-3 ml-2">
                        <button
                            onClick={toggleTheme}
                            className="size-10 sm:size-11 md:size-12 rounded-full bg-surface border border-white/5 flex items-center justify-center text-secondary hover:text-primary transition-colors tap-target"
                            aria-label="Toggle theme"
                        >
                            {theme === 'dark' ? <Sun size={18} className="sm:w-5 sm:h-5" /> : <Moon size={18} className="sm:w-5 sm:h-5" />}
                        </button>
                    </div>
                </header>

                {/* Main Content Area - Responsive Padding */}
                <main className="flex-1 overflow-x-hidden overflow-y-auto smooth-scroll">
                    <div className="w-full max-w-2xl lg:max-w-7xl mx-auto p-3 sm:p-4 md:p-6 pb-24 sm:pb-28 md:pb-10 transition-all duration-300">
                        <Outlet />
                    </div>
                </main>

                {/* Mobile/Consumer Bottom Navigation - Hidden on Desktop */}
                <nav className={clsx(
                    "lg:hidden flex-none bg-surface/80 backdrop-blur-lg border-t border-white/5 z-50",
                    "h-16 sm:h-18 md:h-20", // Responsive height
                    "px-2 sm:px-4 md:px-6", // Responsive padding
                    "flex items-center justify-between",
                    "md:max-w-md md:mx-auto md:mb-6 md:rounded-full md:border md:absolute md:bottom-0 md:left-0 md:right-0"
                )}>
                    <NavItem to="/" icon={Home} label="Home" />
                    <NavItem to="/vitals" icon={Activity} label="Vitals" />
                    <NavItem to="/guidance" icon={Shield} label="Guide" />
                    <NavItem to="/history" icon={History} label="History" />
                    <NavItem to="/settings" icon={Settings} label="Profile" />
                </nav>

                <ChatAssistant />
            </div>
        </div>
    );
}

