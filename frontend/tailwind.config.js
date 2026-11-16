/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        ink: { 950: "#070a10", 900: "#0b1019", 850: "#0f1622", 800: "#141c2b", 700: "#1d2738", 600: "#2a3649", 500: "#3b4a61" },
        accent: { 300: "#a5b4fc", 400: "#818cf8", 500: "#6366f1", 600: "#4f46e5" },
        cyanx: { 300: "#67e8f9", 400: "#22d3ee" },
      },
      boxShadow: { card: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)" },
    },
  },
  plugins: [],
};
