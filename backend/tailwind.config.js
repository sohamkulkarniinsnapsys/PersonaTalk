/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './templates/**/*.html',
        './static/js/**/*.js',
    ],
    theme: {
        extend: {
            colors: {
                'bg-deep': '#0f172a',
                'bg-card': '#1e293b',
                'bg-input': '#334155',
                primary: {
                    DEFAULT: '#6366f1',
                    hover: '#4f46e5',
                    glow: 'rgba(99, 102, 241, 0.5)',
                },
                accent: {
                    DEFAULT: '#14b8a6',
                    hover: '#0d9488',
                },
                danger: '#ef4444',
                success: '#10b981',
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            animation: {
                entrance: 'fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0', transform: 'translateY(10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
            },
        },
    },
    plugins: [],
}
