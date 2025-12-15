import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './layouts/DashboardLayout';
import { ThemeProvider } from './contexts/ThemeContext';

import { LiveMonitor } from './pages/LiveMonitor';
import { Vitals } from './pages/Vitals';
import { Guidance } from './pages/Guidance';
import { History } from './pages/History';
import { Settings } from './pages/Settings';
import { CalendarView } from './pages/CalendarView';
import { SleepQuality } from './pages/SleepQuality';

function App() {
  return (
    <ThemeProvider>
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
    </ThemeProvider>
  );
}

export default App;
