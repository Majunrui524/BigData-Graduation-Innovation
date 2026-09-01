import { AnimatePresence, motion } from "framer-motion";
import { NavLink, useLocation } from "react-router-dom";
import type { PropsWithChildren } from "react";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/detective", label: "Detective" },
  { to: "/graph", label: "Graph" },
  { to: "/communities", label: "Communities" },
  { to: "/compare", label: "Compare" },
  { to: "/errors", label: "Errors" },
];

export function AppShell({ children }: PropsWithChildren) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-paper text-charcoal">
      <div
        className="fixed inset-0 -z-10 opacity-40"
        style={{
          backgroundImage: `url(${import.meta.env.BASE_URL}visuals/paper-noise.png)`,
          backgroundSize: "420px 420px",
        }}
      />
      <header className="sticky top-0 z-30 border-b border-charcoal/10 bg-paper/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-6 px-5 py-4 lg:px-8">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-charcoal/55">
              Interactive Research Demo
            </p>
            <h1 className="font-display text-2xl font-bold tracking-tight">
              Late Fusion Community Structure Explorer
            </h1>
          </div>
        </div>
        <nav className="mx-auto flex max-w-[1440px] gap-2 overflow-auto px-5 pb-4 lg:px-8">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-full border px-4 py-2 font-body text-sm transition ${
                  isActive
                    ? "border-charcoal bg-charcoal text-paper"
                    : "border-charcoal/10 bg-white/70 text-charcoal/70 hover:border-charcoal/40 hover:text-charcoal"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-[1440px] px-5 py-6 lg:px-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.28, ease: "easeOut" }}
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
