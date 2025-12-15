import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './layouts/DashboardLayout';
import { ThemeProvider } from './contexts/ThemeContext';
import { UserProvider } from './contexts/UserContext';
import { ErrorBoundary } from './components/ErrorBoundary';

import { LiveMonitor } from './pages/LiveMonitor';
import { Vitals } from './pages/Vitals';
import { Guidance } from './pages/Guidance';
import { History } from './pages/History';
import { Settings } from './pages/Settings';
import { CalendarView } from './pages/CalendarView';
import { SleepQuality } from './pages/SleepQuality';

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <UserProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<DashboardLayout />}>
                <Route index element={<LiveMonitor />} />
                <Route path="vitals" element={<Vitals />} />
                <Route path="guidance" element={<Guidance />} />
                <Route path="history" element={<History />} />
                <Route path="calendar" element={<CalendarView />} />
                <Route path="sleep" element={<SleepQuality />} />
                <Route path="settings" element={<Settings />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </UserProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
