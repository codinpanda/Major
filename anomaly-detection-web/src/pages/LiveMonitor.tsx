import { useEffect, useState } from 'react';
import { Heart, Activity, Zap, Moon } from 'lucide-react';
import { VitalsCard } from '../components/VitalsCard';
import { RealTimeChart } from '../components/RealTimeChart';
import { GuidanceOverlay } from '../components/GuidanceOverlay';
import { HealthStatus } from '../components/HealthStatus';
import { useUser } from '../contexts/UserContext';
import { dataGenerator } from '../simulation/DataGenerator';
import type { HealthDataPacket } from '../simulation/DataGenerator';
import { inferenceEngine } from '../engine/InferenceEngine';

import { ExportMenu } from '../components/ExportMenu';

export function LiveMonitor() {
    const { user } = useUser();
    const [dataHistory, setDataHistory] = useState<HealthDataPacket[]>([]);
    const [latest, setLatest] = useState<HealthDataPacket | null>(null);
    const [prediction, setPrediction] = useState<number>(0); // Store ML prediction
    const [zScore, setZScore] = useState<number>(0);
    const [isCalibrating, setIsCalibrating] = useState(true);
    const [isSimulating, setIsSimulating] = useState(false);

    useEffect(() => {
        // Init happens in constructor, but we'll monitor status
        const unsubscribe = dataGenerator.subscribe(async (packet) => {
            setLatest(packet);

            // Only predict if data is available (Live or Sim)
            if (packet.rawECG && packet.rawECG.length > 0) {
                const result = await inferenceEngine.predict(packet);
                setPrediction(result.probability);
                setZScore(result.zScore);
            }

            setIsCalibrating(inferenceEngine.getStatus().isCalibrating);

            setDataHistory(prev => {
                const newVal = [...prev, packet];
                if (newVal.length > 60) newVal.shift();
                return newVal;
            });
        });
        return () => { unsubscribe(); };
    }, []);

    const startSim = (mode: 'NORMAL' | 'ANOMALY' | 'RANDOM' | 'EXTERNAL_DEVICE') => {
        dataGenerator.setMode(mode);
        dataGenerator.start();
        setIsSimulating(true);
    };

    const stopSim = () => {
        dataGenerator.stop();
        setIsSimulating(false);
    };

    // Generate friendly relative time labels (e.g., "60s ago", "Now")
    const labels = dataHistory.map((_, i) => {
        const secondsAgo = dataHistory.length - 1 - i;
        return secondsAgo === 0 ? 'Now' : `${secondsAgo}s ago`;
    });
    const hrData = dataHistory.map(d => d.heartRate);

    // Get current time greeting
    const hour = new Date().getHours();
    const greeting = hour < 12 ? 'Good Morning' : hour < 18 ? 'Good Afternoon' : 'Good Evening';
    const today = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });

    return (
        <div className="space-y-6 w-full max-w-7xl mx-auto">
            {/* Welcome Hero */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div>
                    <h1 className="text-3xl lg:text-4xl font-bold text-primary tracking-tight">
                        {greeting}, <span className="text-accent">{user.name}</span>
                    </h1>
                    <p className="text-secondary text-lg mt-1">{today}</p>
                </div>

                {/* Simulation Control Center */}
                <div className="flex items-center gap-2 bg-surface p-1.5 rounded-full border border-white/5 shadow-sm">
                    {/* Calibration Status */}
                    {isCalibrating && isSimulating && (
                        <div className="px-4 flex items-center gap-2 text-sm font-medium text-warning animate-pulse">
                            <span className="size-2 rounded-full bg-warning"></span>
                            Calibrating...
                        </div>
                    )}

                    {!isCalibrating && isSimulating && (
                        <div className="px-4 hidden md:flex items-center gap-2 text-sm font-medium text-secondary border-r border-white/10 pr-4 mr-2">
                            <span>Confidence:</span>
                            <span className={zScore > 2.5 ? 'text-danger font-bold' : 'text-success'}>
                                {zScore.toFixed(1)}σ
                            </span>
                        </div>
                    )}

                    {!isSimulating ? (
                        <button
                            onClick={() => startSim('NORMAL')}
                            className="text-sm font-medium px-4 py-2 bg-accent text-white rounded-full shadow-lg shadow-accent/20 hover:scale-105 transition-transform"
                        >
                            Start Live Demo
                        </button>
                    ) : (
                        <>
                            <div className="px-4 flex items-center gap-2 text-sm font-medium text-success">
                                <span className="relative flex h-2 w-2">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
                                </span>
                                Monitoring Active
                            </div>
                            <div className="h-6 w-px bg-white/10 mx-1" />
                            <button
                                onClick={() => startSim('EXTERNAL_DEVICE')}
                                className="text-sm font-medium px-4 py-2 hover:bg-white/5 text-secondary hover:text-accent hover:bg-accent/10 rounded-full transition-colors"
                            >
                                Connect Device
                            </button>
                            <button
                                onClick={() => startSim('ANOMALY')}
                                className="text-sm font-medium px-4 py-2 hover:bg-white/5 text-secondary hover:text-danger hover:bg-danger/10 rounded-full transition-colors"
                            >
                                Trigger Anomaly
                            </button>
                            <button
                                onClick={stopSim}
                                className="text-sm font-medium px-4 py-2 hover:bg-white/5 text-secondary hover:text-primary rounded-full transition-colors"
                            >
                                Stop
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* Health Status Card */}
            <HealthStatus latest={latest} anomalyScore={prediction} />

            {/* Metric Cards - Responsive Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
                <VitalsCard
                    title="Heart Rate"
                    value={latest?.heartRate?.toFixed(0) ?? '--'}
                    unit="BPM"
                    icon={Heart}
                    status={latest?.heartRate && latest.heartRate > 100 ? "critical" : "normal"}
                />
                <VitalsCard
                    title="Heart Rate Variability"
                    value={latest?.hrv?.toFixed(0) ?? '--'}
                    unit="ms"
                    icon={Activity}
                    status="normal"
                />
                <VitalsCard
                    title="Stress Level"
                    value={latest?.stress?.toFixed(0) ?? '--'}
                    unit="%"
                    icon={Zap}
                    status={latest?.stress && latest.stress > 60 ? "warning" : "normal"}
                />
                <VitalsCard
                    title="Sleep Quality"
                    value="7h 32m"
                    unit="last night"
                    icon={Moon}
                    status="normal"
                />
            </div>

            {/* Heart Rate Trend Chart - Responsive Height */}
            <div className="card-base p-5 lg:p-6 bg-surface">
                <div className="flex justify-between items-center mb-4 lg:mb-6">
                    <div className="flex items-center gap-3">
                        <div className="size-10 rounded-xl bg-accent/10 text-accent flex items-center justify-center">
                            <Activity size={20} />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-primary">Heart Rate Trend</h3>
                            <p className="text-xs text-secondary">Real-time monitoring</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="px-3 py-1 rounded-full bg-background border border-white/5 text-xs font-medium text-secondary">
                            Last Hour
                        </div>
                        <ExportMenu
                            chartId="hr-chart"
                            chartTitle={`Health Data - ${new Date().toLocaleDateString()}`}
                            data={dataHistory}
                            headers={['timestamp', 'heartRate', 'hrv', 'stress']}
                        />
                    </div>
                </div>
                <div className="h-64 lg:h-96 transition-all duration-300" id="hr-chart">
                    <RealTimeChart label="Heart Rate" data={hrData} labels={labels} color="#00C7BE" min={40} max={160} normalMin={60} normalMax={100} />
                </div>
            </div>

            <GuidanceOverlay latestData={latest} anomalyScore={prediction} />
        </div>
    );
}
