# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-12-14

### 🎉 Initial Release - Samsung Health UI Redesign

#### Added
- **Samsung Health-style UI** with teal accent colors and card-based layout
- **5 Main Pages**:
  - Home (LiveMonitor) - Real-time health dashboard
  - Vitals - Interactive data dashboard with charts
  - Guidance - Wellness tips and breathing exercises
  - History - Progress tracking
  - Settings - Profile and preferences
- **Dark/Light Theme Toggle** with persistent storage
- **Bottom Navigation** with 5 icons (Home, Vitals, Guide, History, Profile)
- **Micro-Animations**: fadeIn, slideUp, pulseSlow, countUp
- **Loading States**: Skeleton loaders for cards and charts
- **Empty/Error States**: Friendly messaging with retry buttons
- **Sparkline Component**: Mini trend charts for metrics
- **Theme Context**: Global theme state management
- **Responsive Design**: Mobile-first with single-column layout

#### Components
- `VitalsCard` - Samsung Health-style metric cards
- `HealthStatus` - Hero card with AI insights
- `RealTimeChart` - Interactive Chart.js charts with normal range highlighting
- `Sparkline` - SVG-based mini trend charts
- `SkeletonLoader` - Loading placeholders (Card, Chart)
- `EmptyState` / `ErrorState` - User-friendly states
- `ChatAssistant` - Floating chat button
- `GuidanceOverlay` - Anomaly alert modal

#### Features
- **Vitals Dashboard**:
  - 4 metric selectors (Heart Rate, Stress, SpO₂, Sleep)
  - 5 time ranges (1H, 6H, 1D, 1W, 1M)
  - Stats summary (Average, Peak, Lowest)
  - Trend analysis
  - Smart insights
- **Guidance Page**:
  - Quick action buttons
  - Breathing exercises (Box, 4-7-8)
  - Health tips
  - Emergency contact card
- **Settings Page**:
  - Profile card with stats
  - Personal information
  - Notification preferences
  - Health goals with progress bars
  - Alert thresholds

#### Technical
- **Dependencies Installed**:
  - framer-motion (animations)
  - recharts (sparklines)
  - react-calendar (calendar view)
  - html2canvas (chart export)
  - jspdf (PDF generation)
- **TypeScript**: Strict mode enabled
- **Tailwind CSS**: Custom color palette
- **Vite**: Fast HMR and build
- **Chart.js**: Interactive visualizations

#### Documentation
- Comprehensive README.md
- Component documentation (COMPONENTS.md)
- Development guide (DEVELOPMENT.md)
- Sprint 1 progress report

---

## [0.2.0] - 2025-12-13

### Added
- Consumer-friendly UI redesign
- Chatbot assistant integration
- Emergency escalation trigger
- Simulation modes (Normal, Anomaly, Random)

---

## [0.1.0] - 2025-12-12

### Added
- Initial project setup
- Hybrid LSTM-GRU model
- ONNX Runtime integration
- Basic dashboard layout
- Real-time data simulation
- Anomaly detection engine

---

## Upcoming Features

### Sprint 2 (Planned)
- [ ] Interactive breathing exercise with animated circle
- [ ] Smart insights engine with pattern detection
- [ ] Achievement badges system
- [ ] Calendar view with color-coded days

### Sprint 3 (Planned)
- [ ] Chart export (PNG, PDF, CSV)
- [ ] Profile photo upload
- [ ] Customizable metric order
- [ ] Multi-metric comparison charts

### Sprint 4 (Planned)
- [ ] Full accessibility support (ARIA, keyboard nav)
- [ ] Performance optimizations (lazy loading, memoization)
- [ ] Onboarding tour for new users
- [ ] High contrast mode

---

## Breaking Changes

### v1.0.0
- Complete UI overhaul - old components deprecated
- New routing structure
- Theme context required

---

## Migration Guide

### From v0.2.0 to v1.0.0

1. **Update imports**:
```tsx
// Old
import { VitalsCard } from './components/VitalsCard';

// New (same, but props changed)
<VitalsCard title="HR" value="72" unit="BPM" status="normal" icon={Heart} />
```

2. **Wrap App with ThemeProvider**:
```tsx
import { ThemeProvider } from './contexts/ThemeContext';

<ThemeProvider>
  <App />
</ThemeProvider>
```

3. **Update routes**:
```tsx
// New route added
<Route path="vitals" element={<Vitals />} />
```

---

## Contributors

- Priyanshu (@priyanshu) - Lead Developer
- Your teammates here

---

**Questions?** See [README.md](./README.md) or open an issue!
