/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class', // Enable class-based dark mode
    theme: {
        // Mobile-first responsive breakpoints
        screens: {
            'xs': '475px',
            'sm': '640px',
            'md': '768px',
            'lg': '1024px',
            'xl': '1280px',
            '2xl': '1536px',
        },
        extend: {
            colors: {
                // Dark theme colors (default)
                background: '#1C1C1E',
                surface: '#2C2C2E',
                surfaceHover: '#3A3A3C',
                primary: '#F2F2F7',
                secondary: '#98989D',
                accent: '#00C7BE',
                accentDark: '#00A89C',
                accentLight: '#00E5D9',
                success: '#30D158',
                successDark: '#28B04A',
                warning: '#FFD60A',
                warningDark: '#F5C000',
                danger: '#FF453A',
                dangerDark: '#E63B30',
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            // Responsive spacing
            spacing: {
                'safe-top': 'env(safe-area-inset-top)',
                'safe-bottom': 'env(safe-area-inset-bottom)',
                'safe-left': 'env(safe-area-inset-left)',
                'safe-right': 'env(safe-area-inset-right)',
            },
            // Container max-widths for different breakpoints
            maxWidth: {
                'mobile': '100%',
                'tablet': '768px',
                'desktop': '1280px',
            },
            // Fluid typography
            fontSize: {
                'xs': ['0.75rem', { lineHeight: '1rem' }],
                'sm': ['0.875rem', { lineHeight: '1.25rem' }],
                'base': ['1rem', { lineHeight: '1.5rem' }],
                'lg': ['1.125rem', { lineHeight: '1.75rem' }],
                'xl': ['1.25rem', { lineHeight: '1.75rem' }],
                '2xl': ['1.5rem', { lineHeight: '2rem' }],
                '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
                '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
            },
            // Touch-friendly minimum sizes
            minHeight: {
                'touch': '44px',
            },
            minWidth: {
                'touch': '44px',
            },
        },
    },
    plugins: [],
}
