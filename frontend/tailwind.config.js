/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#172033",
        muted: "#64748b",
        line: "#d8dee9",
        panel: "#ffffff",
        canvas: "#f4f6f8",
        accent: "#2356a8",
      },
    },
  },
  plugins: [],
};
