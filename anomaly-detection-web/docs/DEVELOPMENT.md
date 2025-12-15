# Development Guide

## Quick Start for New Team Members

### 1. Clone and Setup
```bash
git clone https://github.com/your-org/anomaly-detection-web.git
cd anomaly-detection-web
npm install
npm run dev
```

### 2. Open in Browser
Navigate to `http://localhost:5173/`

### 3. Explore the App
- **Home** (`/`) - Live monitoring dashboard
- **Vitals** (`/vitals`) - Detailed metrics with charts
- **Guide** (`/guidance`) - Wellness tips
- **History** (`/history`) - Progress tracking
- **Profile** (`/settings`) - Settings and preferences

---

## Development Workflow

### Daily Workflow

1. **Pull latest changes**
```bash
git pull origin main
```

2. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Start dev server**
```bash
npm run dev
```

4. **Make changes** and test in browser (HMR enabled)

5. **Commit and push**
```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

6. **Create Pull Request** on GitHub

---

## Project Architecture

### Tech Stack Overview

```
Frontend (Browser)
├── React 18.3 (UI)
├── TypeScript 5.6 (Type Safety)
├── Tailwind CSS (Styling)
├── Chart.js (Visualizations)
└── ONNX Runtime (ML Inference)

Build Tools
├── Vite (Dev Server + Bundler)
└── ESLint (Linting)

Backend (Future)
└── Node.js/Python API (TBD)
```

### Data Flow

```
User Interaction
    ↓
React Component
    ↓
DataGenerator (Simulation)
    ↓
InferenceEngine (ML)
    ↓
State Update
    ↓
UI Re-render
```

---

## File Organization

### Adding New Components

1. Create file in `src/components/`
```tsx
// src/components/MyComponent.tsx
export function MyComponent({ prop1, prop2 }: MyComponentProps) {
  return <div>...</div>;
}
```

2. Export from component (optional)
```tsx
// src/components/index.ts
export { MyComponent } from './MyComponent';
```

3. Use in pages
```tsx
import { MyComponent } from '../components/MyComponent';
```

### Adding New Pages

1. Create file in `src/pages/`
```tsx
// src/pages/MyPage.tsx
export function MyPage() {
  return <div>My Page Content</div>;
}
```

2. Add route in `src/App.tsx`
```tsx
<Route path="mypage" element={<MyPage />} />
```

3. Add navigation link in `DashboardLayout.tsx`
```tsx
<NavItem to="/mypage" icon={MyIcon} label="My Page" />
```

---

## Styling Guide

### Tailwind CSS

**Use utility classes**:
```tsx
<div className="card-base p-5 bg-surface">
  <h2 className="text-xl font-bold text-primary">Title</h2>
  <p className="text-sm text-secondary mt-2">Description</p>
</div>
```

**Common patterns**:
```tsx
// Card
className="card-base p-5 bg-surface"

// Button
className="px-4 py-2 bg-accent text-surface rounded-xl font-medium hover:bg-accentDark"

// Icon container
className="size-10 rounded-full bg-accent/10 text-accent flex items-center justify-center"

// Grid (2 columns)
className="grid grid-cols-2 gap-3"

// Vertical spacing
className="space-y-4"
```

### Custom Colors

**Access in Tailwind**:
```tsx
bg-background    // #1C1C1E
bg-surface       // #2C2C2E
text-primary     // #F2F2F7
text-secondary   // #98989D
bg-accent        // #00C7BE
text-success     // #30D158
text-warning     // #FFD60A
text-danger      // #FF453A
```

**Opacity modifiers**:
```tsx
bg-accent/10     // 10% opacity
bg-accent/20     // 20% opacity
border-accent/30 // 30% opacity
```

---

## State Management

### Local State (useState)
```tsx
const [count, setCount] = useState(0);
```

### Context (Global State)
```tsx
// Create context
const MyContext = createContext<MyContextType | undefined>(undefined);

// Provider
export function MyProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState(initialValue);
  return <MyContext.Provider value={{ value, setValue }}>{children}</MyContext.Provider>;
}

// Hook
export function useMyContext() {
  const context = useContext(MyContext);
  if (!context) throw new Error('useMyContext must be used within MyProvider');
  return context;
}
```

### Example: Theme Context
```tsx
import { useTheme } from './contexts/ThemeContext';

function MyComponent() {
  const { theme, toggleTheme } = useTheme();
  return <button onClick={toggleTheme}>{theme}</button>;
}
```

---

## Working with Data

### Simulated Data
```tsx
import { DataGenerator } from '../simulation/DataGenerator';

const generator = new DataGenerator();
generator.start((data) => {
  console.log(data); // { hr, hrv, spo2, motion, timestamp }
});
```

### ML Inference
```tsx
import { InferenceEngine } from '../engine/InferenceEngine';

const engine = new InferenceEngine();
await engine.initialize();

const result = await engine.predict(dataSequence);
console.log(result); // { isAnomaly: boolean, probability: number }
```

---

## Common Tasks

### Adding a New Metric

1. **Update DataGenerator** (`src/simulation/DataGenerator.ts`)
```tsx
emit() {
  const packet = {
    // ... existing metrics
    newMetric: this.generateNewMetric(),
  };
}
```

2. **Update Schema** (`src/types/schema.ts`)
```tsx
export const HealthDataSchema = z.object({
  // ... existing fields
  newMetric: z.number(),
});
```

3. **Add VitalsCard** (`src/pages/LiveMonitor.tsx`)
```tsx
<VitalsCard
  title="New Metric"
  value={data.newMetric.toFixed(0)}
  unit="UNIT"
  status="normal"
  icon={MyIcon}
/>
```

### Adding a Chart

1. **Import RealTimeChart**
```tsx
import { RealTimeChart } from '../components/RealTimeChart';
```

2. **Prepare data**
```tsx
const [chartData, setChartData] = useState<number[]>([]);
const [labels, setLabels] = useState<string[]>([]);
```

3. **Render chart**
```tsx
<RealTimeChart
  label="My Metric"
  data={chartData}
  labels={labels}
  color="#00C7BE"
  min={0}
  max={100}
/>
```

### Adding an Animation

1. **Use existing classes**
```tsx
<div className="animate-fade-in">Content</div>
```

2. **Or create custom keyframe** (`src/index.css`)
```css
@keyframes myAnimation {
  from { opacity: 0; }
  to { opacity: 1; }
}

.animate-my-animation {
  animation: myAnimation 1s ease-in-out;
}
```

---

## Debugging

### React DevTools
1. Install [React DevTools](https://react.dev/learn/react-developer-tools)
2. Open browser DevTools → React tab
3. Inspect component tree and props

### Console Logging
```tsx
console.log('Data:', data);
console.table(data); // For arrays/objects
console.error('Error:', error);
```

### TypeScript Errors
```bash
npm run type-check
```

### Lint Errors
```bash
npm run lint
```

---

## Testing

### Manual Testing Checklist
- [ ] Home page loads
- [ ] Vitals dashboard shows charts
- [ ] Theme toggle works
- [ ] Navigation works (all 5 pages)
- [ ] Responsive on mobile (resize browser)
- [ ] No console errors

### Browser Testing
- Chrome (primary)
- Firefox
- Safari
- Edge

### Mobile Testing
- Use browser DevTools → Toggle device toolbar
- Test on actual device if possible

---

## Performance

### Vite HMR (Hot Module Replacement)
- Changes auto-reload in browser
- State preserved when possible
- Fast refresh (<100ms)

### Build Optimization
```bash
npm run build
```
- Minification
- Tree-shaking
- Code splitting
- Asset optimization

### Performance Tips
1. Use `React.memo()` for expensive components
2. Lazy load heavy components
3. Debounce frequent updates
4. Optimize images (WebP, compression)

---

## Git Best Practices

### Commit Messages
```bash
feat: add new vitals card
fix: resolve chart rendering bug
docs: update README
style: format code
refactor: simplify data generator
test: add unit tests
chore: update dependencies
```

### Branch Naming
```bash
feature/add-calendar-view
fix/chart-rendering-bug
docs/update-readme
refactor/simplify-state
```

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactor

## Testing
- [ ] Tested locally
- [ ] No console errors
- [ ] Responsive design verified

## Screenshots
(if applicable)
```

---

## Troubleshooting

### Port Already in Use
```bash
npx kill-port 5173
npm run dev
```

### Module Not Found
```bash
rm -rf node_modules
npm install
```

### TypeScript Errors
```bash
# Clear cache
rm -rf node_modules/.vite
npm run dev
```

### Build Fails
```bash
# Check for errors
npm run type-check
npm run lint

# Clean build
rm -rf dist
npm run build
```

---

## Resources

### Documentation
- [React Docs](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Vite Guide](https://vite.dev/guide/)
- [Chart.js Docs](https://www.chartjs.org/docs/)

### Tools
- [React DevTools](https://react.dev/learn/react-developer-tools)
- [VS Code](https://code.visualstudio.com/)
- [Postman](https://www.postman.com/) (for API testing)

### Community
- [React Discord](https://discord.gg/react)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/reactjs)

---

## Getting Help

1. **Check documentation** (README, COMPONENTS.md)
2. **Search existing issues** on GitHub
3. **Ask in team chat** (Slack/Discord)
4. **Create an issue** with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots/logs

---

**Happy coding! 🚀**
