# Antigravity Health Monitoring App

A **Samsung Health-style** web application for real-time health monitoring with AI-powered anomaly detection. Built with React, TypeScript, and Vite.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![React](https://img.shields.io/badge/React-18.3-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178c6)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development](#development)
- [UI Components](#ui-components)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## ✨ Features

### 🏠 **Home Dashboard**
- Personalized greeting with live monitoring status
- Health Status hero card with AI insights
- Real-time vitals cards (Heart Rate, HRV, Stress, Sleep)
- Interactive trend charts
- Demo simulation controls

### 📊 **Vitals Dashboard**
- 4 metric selectors (Heart Rate, Stress, SpO₂, Sleep)
- Time range selector (1H, 6H, 1D, 1W, 1M)
- Stats summary (Average, Peak, Lowest)
- Interactive charts with normal range highlighting
- Smart insights and trend analysis

### 🧘 **Wellness Guidance**
- Quick action buttons (Breathing, Activity logging)
- Categorized breathing exercises
- Health tips and stress management
- Emergency contact card

### 📅 **Health History**
- Today's summary with stats grid
- Weekly trends with improvement indicators
- Recent activity log

### ⚙️ **Profile & Settings**
- Profile card with health stats
- Personal information management
- Notification preferences
- Health goals with progress bars
- Alert threshold configuration

### 🎨 **UI/UX Features**
- **Samsung Health Design**: Teal accent, card-based layout
- **Dark/Light Theme**: Toggle in header
- **Animations**: Fade-in, slide-up, pulse effects
- **Loading States**: Skeleton loaders
- **Empty/Error States**: Friendly messaging
- **Mobile-First**: Responsive design
- **Bottom Navigation**: 5-icon navigation bar

---

## 🛠 Tech Stack

### Frontend
- **React 18.3** - UI library
- **TypeScript 5.6** - Type safety
- **Vite 5.4** - Build tool & dev server
- **Tailwind CSS 3.4** - Utility-first styling
- **React Router 7.1** - Client-side routing

### Charts & Visualization
- **Chart.js 4.4** - Interactive charts
- **Recharts** - Sparklines
- **chartjs-plugin-zoom** - Chart interactions
- **chartjs-plugin-annotation** - Range highlighting

### AI/ML
- **ONNX Runtime Web 1.20** - Browser-based ML inference
- **PyTorch** - Model training (Python)

### Utilities
- **Zod 3.24** - Schema validation
- **Lucide React** - Icon library
- **clsx** - Conditional classes
- **Framer Motion** - Advanced animations
- **html2canvas** - Chart export
- **jsPDF** - PDF generation

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+ and npm
- **Python 3.8+** (for ML model training)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/anomaly-detection-web.git
cd anomaly-detection-web
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm run dev
```

The app will be available at `http://localhost:5173/`

### Build for Production
```bash
npm run build
npm run preview  # Preview production build
```

---

## 📁 Project Structure

```
anomaly-detection-web/
├── public/                    # Static assets
│   ├── model.onnx            # ML model
│   └── confusion_matrix.svg  # Model metrics
├── src/
│   ├── components/           # Reusable UI components
│   │   ├── ChatAssistant.tsx
│   │   ├── EmptyState.tsx
│   │   ├── GuidanceOverlay.tsx
│   │   ├── HealthStatus.tsx
│   │   ├── RealTimeChart.tsx
│   │   ├── SkeletonLoader.tsx
│   │   ├── Sparkline.tsx
│   │   └── VitalsCard.tsx
│   ├── contexts/             # React contexts
│   │   └── ThemeContext.tsx  # Dark/Light theme
│   ├── engine/               # ML inference
│   │   └── InferenceEngine.ts
│   ├── layouts/              # Page layouts
│   │   └── DashboardLayout.tsx
│   ├── pages/                # Route pages
│   │   ├── Guidance.tsx
│   │   ├── History.tsx
│   │   ├── LiveMonitor.tsx   # Home page
│   │   ├── Settings.tsx
│   │   └── Vitals.tsx
│   ├── simulation/           # Data simulation
│   │   └── DataGenerator.ts
│   ├── types/                # TypeScript types
│   │   └── schema.ts
│   ├── App.tsx               # Root component
│   ├── index.css             # Global styles
│   └── main.tsx              # Entry point
├── scripts/                  # Utility scripts
│   ├── train_model.py        # ML model training
│   └── generate_artifacts.js # Generate assets
├── notebooks/                # Jupyter notebooks
│   └── model_training.ipynb
├── package.json
├── tailwind.config.js        # Tailwind configuration
├── tsconfig.json             # TypeScript config
└── vite.config.ts            # Vite config
```

---

## 💻 Development

### Available Scripts

```bash
npm run dev          # Start dev server with HMR
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript compiler check
```

### Environment Setup

No environment variables required for basic development. The app uses:
- Local simulation for health data
- Client-side ML inference (ONNX)
- LocalStorage for theme preference

### Code Style

- **TypeScript**: Strict mode enabled
- **ESLint**: Configured for React + TypeScript
- **Prettier**: (Optional) Add `.prettierrc` for formatting
- **Naming**: PascalCase for components, camelCase for functions

---

## 🧩 UI Components

### Core Components

#### `VitalsCard`
Displays a single health metric with icon, value, and interpretation.
```tsx
<VitalsCard
  title="Heart Rate"
  value="72"
  unit="BPM"
  status="normal"
  icon={Heart}
/>
```

#### `Sparkline`
Mini trend chart for metric cards.
```tsx
<Sparkline 
  data={[72, 75, 73, 78, 74]} 
  color="#00C7BE" 
  width={60} 
  height={24} 
/>
```

#### `SkeletonLoader`
Loading placeholder.
```tsx
{loading ? <CardSkeleton /> : <VitalsCard {...props} />}
```

#### `EmptyState` / `ErrorState`
Friendly messaging for empty or error states.
```tsx
<EmptyState 
  title="No data yet"
  description="Start monitoring to see your vitals"
  action={{ label: "Start", onClick: handleStart }}
/>
```

### Theme Context

```tsx
import { useTheme } from './contexts/ThemeContext';

function MyComponent() {
  const { theme, toggleTheme } = useTheme();
  return <button onClick={toggleTheme}>Toggle</button>;
}
```

---

## 🎨 Styling Guide

### Color Palette

**Dark Theme** (Default):
```javascript
background: '#1C1C1E'    // Soft dark gray
surface: '#2C2C2E'       // Card background
accent: '#00C7BE'        // Teal (Samsung Health)
success: '#30D158'       // Green
warning: '#FFD60A'       // Yellow
danger: '#FF453A'        // Red
```

**Light Theme**:
```javascript
background: '#F5F5F7'
surface: '#FFFFFF'
primary: '#1C1C1E'
secondary: '#8E8E93'
```

### Animation Classes

```css
.animate-fade-in      /* Fade in on mount */
.animate-slide-up     /* Slide up from bottom */
.animate-pulse-slow   /* Gentle pulse */
.animate-count-up     /* Number counting */
```

### Card Utilities

```css
.card-base            /* Base card style */
.card-hover           /* Hover effect */
```

---

## 🚢 Deployment

### Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

### Netlify
```bash
npm run build
# Drag 'dist' folder to Netlify
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

---

## 🤝 Contributing

### Workflow

1. Create a feature branch
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and commit
```bash
git add .
git commit -m "feat: add new feature"
```

3. Push and create PR
```bash
git push origin feature/your-feature-name
```

### Commit Convention

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Tests
- `chore:` Maintenance

---

## 📚 Additional Documentation

- [Component Documentation](./docs/COMPONENTS.md)
- [API Reference](./docs/API.md)
- [ML Model Guide](./docs/ML_MODEL.md)
- [Sprint Progress](./docs/SPRINTS.md)

---

## 🐛 Troubleshooting

### Port already in use
```bash
# Kill process on port 5173
npx kill-port 5173
npm run dev
```

### Build errors
```bash
# Clear cache and reinstall
rm -rf node_modules dist
npm install
npm run build
```

### TypeScript errors
```bash
# Check types
npm run type-check
```

---

## 📄 License

MIT License - see [LICENSE](./LICENSE) file

---

## 👥 Team

- **Priyanshu** - Lead Developer
- Your teammates here

---

## 🙏 Acknowledgments

- Samsung Health for design inspiration
- WESAD dataset for health data simulation
- React and Vite communities

---

**Questions?** Open an issue or contact the team!
