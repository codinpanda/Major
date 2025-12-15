# Component Documentation

## Overview
This document provides detailed information about all UI components in the Antigravity Health Monitoring App.

---

## Core Components

### `VitalsCard`
**Location**: `src/components/VitalsCard.tsx`

**Purpose**: Display a single health metric with Samsung Health styling.

**Props**:
```typescript
interface VitalsCardProps {
    title: string;           // Metric name (e.g., "Heart Rate")
    value: string;           // Current value (e.g., "72")
    unit: string;            // Unit (e.g., "BPM")
    status?: 'normal' | 'warning' | 'critical';
    icon?: React.ElementType; // Lucide icon component
}
```

**Usage**:
```tsx
import { Heart } from 'lucide-react';
import { VitalsCard } from './components/VitalsCard';

<VitalsCard
  title="Heart Rate"
  value="72"
  unit="BPM"
  status="normal"
  icon={Heart}
/>
```

**Styling**:
- Icon-left layout
- Color-coded by status (teal/yellow/red)
- Rounded background for icon
- Big number display (text-4xl)

---

### `Sparkline`
**Location**: `src/components/Sparkline.tsx`

**Purpose**: Lightweight SVG mini-chart for showing trends.

**Props**:
```typescript
interface SparklineProps {
    data: number[];          // Array of data points
    color?: string;          // Line color (default: '#00C7BE')
    width?: number;          // SVG width (default: 60)
    height?: number;         // SVG height (default: 24)
}
```

**Usage**:
```tsx
<Sparkline 
  data={[72, 75, 73, 78, 74, 76]} 
  color="#00C7BE" 
  width={60} 
  height={24} 
/>
```

**Features**:
- Auto-scales to data range
- Smooth polyline rendering
- Minimal performance overhead

---

### `HealthStatus`
**Location**: `src/components/HealthStatus.tsx`

**Purpose**: Hero card showing overall health status.

**Props**:
```typescript
interface HealthStatusProps {
    status: 'good' | 'attention' | 'critical';
    message: string;
    subMessage?: string;
}
```

**Usage**:
```tsx
<HealthStatus 
  status="good"
  message="You're doing great today"
  subMessage="Most metrics look normal"
/>
```

---

### `RealTimeChart`
**Location**: `src/components/RealTimeChart.tsx`

**Purpose**: Interactive Chart.js line chart with normal range highlighting.

**Props**:
```typescript
interface RealTimeChartProps {
    label: string;
    data: number[];
    labels: string[];
    color: string;
    min?: number;
    max?: number;
    normalMin?: number;
    normalMax?: number;
}
```

**Usage**:
```tsx
<RealTimeChart
  label="Heart Rate"
  data={hrData}
  labels={timeLabels}
  color="#00C7BE"
  min={40}
  max={160}
  normalMin={60}
  normalMax={100}
/>
```

**Features**:
- Smooth line rendering
- Normal range shading (green band)
- Responsive height
- Tooltip on hover

---

### `SkeletonLoader`
**Location**: `src/components/SkeletonLoader.tsx`

**Purpose**: Loading placeholders for better UX.

**Components**:
1. `SkeletonLoader` - Generic skeleton
2. `CardSkeleton` - For vitals cards
3. `ChartSkeleton` - For chart areas

**Usage**:
```tsx
import { CardSkeleton, ChartSkeleton } from './components/SkeletonLoader';

{loading ? <CardSkeleton /> : <VitalsCard {...props} />}
{loading ? <ChartSkeleton /> : <RealTimeChart {...props} />}
```

---

### `EmptyState` & `ErrorState`
**Location**: `src/components/EmptyState.tsx`

**Purpose**: Friendly messaging for empty data or errors.

**Props**:
```typescript
interface EmptyStateProps {
    title: string;
    description: string;
    icon?: React.ElementType;
    action?: {
        label: string;
        onClick: () => void;
    };
}

interface ErrorStateProps {
    message: string;
    onRetry: () => void;
}
```

**Usage**:
```tsx
<EmptyState 
  title="No data yet"
  description="Start monitoring to see your vitals"
  icon={Activity}
  action={{ label: "Start Monitoring", onClick: handleStart }}
/>

<ErrorState 
  message="Failed to load data"
  onRetry={handleRetry}
/>
```

---

### `ChatAssistant`
**Location**: `src/components/ChatAssistant.tsx`

**Purpose**: Floating chat button for AI assistance.

**Features**:
- Fixed position (bottom-right)
- Teal accent color
- Message icon
- Click to open chat

---

### `GuidanceOverlay`
**Location**: `src/components/GuidanceOverlay.tsx`

**Purpose**: Modal overlay for anomaly alerts and guidance.

**Props**:
```typescript
interface GuidanceOverlayProps {
    isOpen: boolean;
    onClose: () => void;
    severity: 'mild' | 'moderate' | 'severe';
    latestData: HealthDataPacket;
}
```

**Usage**:
```tsx
<GuidanceOverlay
  isOpen={showGuidance}
  onClose={() => setShowGuidance(false)}
  severity="moderate"
  latestData={currentData}
/>
```

**Features**:
- Friendly language ("Something looks a bit unusual")
- Severity-based messaging
- Breathing exercise button
- Emergency contact option

---

## Layout Components

### `DashboardLayout`
**Location**: `src/layouts/DashboardLayout.tsx`

**Purpose**: Main app layout with header and bottom navigation.

**Features**:
- Greeting header ("Good Morning, Priyanshu")
- Live monitoring status with pulsing dot
- Theme toggle button (Sun/Moon)
- Settings button
- Bottom navigation (5 icons)
- Outlet for page content

**Navigation Items**:
1. Home (Today)
2. Vitals (Dashboard)
3. Guide (Wellness)
4. History (Progress)
5. Profile (Settings)

---

## Context Providers

### `ThemeContext`
**Location**: `src/contexts/ThemeContext.tsx`

**Purpose**: Global theme state (dark/light mode).

**API**:
```typescript
interface ThemeContextType {
    theme: 'dark' | 'light';
    toggleTheme: () => void;
}
```

**Usage**:
```tsx
import { useTheme } from './contexts/ThemeContext';

function MyComponent() {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <button onClick={toggleTheme}>
      {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
    </button>
  );
}
```

**Features**:
- Persists to localStorage
- Applies `.light` class to `<html>`
- Automatic on mount

---

## Utility Components

### Animation Classes

**Location**: `src/index.css`

**Available Classes**:
```css
.animate-fade-in      /* Fade in (0.5s) */
.animate-slide-up     /* Slide up (0.6s) */
.animate-pulse-slow   /* Pulse (2s infinite) */
.animate-count-up     /* Scale up (1s) */
```

**Usage**:
```tsx
<div className="animate-fade-in">
  <VitalsCard {...props} />
</div>
```

---

## Best Practices

### 1. **Component Composition**
```tsx
// Good: Compose small components
<div className="card-base p-5 bg-surface">
  <Sparkline data={data} />
  <VitalsCard {...props} />
</div>

// Avoid: Monolithic components
```

### 2. **Loading States**
```tsx
// Always show skeleton while loading
{isLoading ? <CardSkeleton /> : <VitalsCard {...props} />}
```

### 3. **Error Handling**
```tsx
// Use ErrorState for user-facing errors
{error ? <ErrorState message={error} onRetry={refetch} /> : <Content />}
```

### 4. **Animations**
```tsx
// Apply to container, not individual elements
<div className="space-y-4 animate-fade-in">
  {items.map(item => <Card key={item.id} {...item} />)}
</div>
```

### 5. **Theme Awareness**
```tsx
// Use Tailwind's dark mode classes if needed
<div className="bg-surface dark:bg-surface light:bg-white">
```

---

## Component Checklist

When creating new components:
- [ ] TypeScript props interface
- [ ] Default props where applicable
- [ ] Responsive design (mobile-first)
- [ ] Accessibility (ARIA labels)
- [ ] Loading state support
- [ ] Error state handling
- [ ] Animation classes
- [ ] Samsung Health styling

---

## Testing Components

```tsx
// Example test structure (if using Vitest/Jest)
import { render, screen } from '@testing-library/react';
import { VitalsCard } from './VitalsCard';

test('renders vitals card', () => {
  render(<VitalsCard title="HR" value="72" unit="BPM" />);
  expect(screen.getByText('72')).toBeInTheDocument();
});
```

---

## Performance Tips

1. **Memoize expensive components**:
```tsx
import { memo } from 'react';
export const VitalsCard = memo(({ ...props }) => { ... });
```

2. **Lazy load charts**:
```tsx
const RealTimeChart = lazy(() => import('./RealTimeChart'));
```

3. **Debounce chart updates**:
```tsx
const debouncedData = useMemo(() => debounce(data, 100), [data]);
```

---

**Questions?** Check the main [README](../README.md) or open an issue!
