export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        extend: {
            colors: {
                ivory: "#f6efe5",
                charcoal: "#111111",
                brick: "#a8402d",
                cobalt: "#183b77",
                acid: "#b2da41",
                ink: "#272521",
                paper: "#faf4eb",
                blush: "#f1d6c6",
            },
            fontFamily: {
                display: ["\"Space Grotesk\"", "ui-sans-serif", "system-ui", "sans-serif"],
                body: ["\"IBM Plex Sans\"", "ui-sans-serif", "system-ui", "sans-serif"],
                mono: ["\"JetBrains Mono\"", "ui-monospace", "SFMono-Regular", "monospace"],
            },
            boxShadow: {
                editorial: "0 16px 40px rgba(17, 17, 17, 0.08)",
            },
            borderRadius: {
                xl2: "1.5rem",
            },
        },
    },
    plugins: [],
};
