import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface MetricCardProps {
  title: string;
  value: string;
  helper?: string;
  accent?: "brick" | "cobalt" | "acid" | "charcoal";
  icon?: ReactNode;
}

const accentClass: Record<NonNullable<MetricCardProps["accent"]>, string> = {
  brick: "text-brick",
  cobalt: "text-cobalt",
  acid: "text-[#6f8f14]",
  charcoal: "text-charcoal",
};

export function MetricCard({ title, value, helper, accent = "charcoal", icon }: MetricCardProps) {
  return (
    <motion.div
      className="metric-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <p className="section-kicker">{title}</p>
        {icon ? <div className={accentClass[accent]}>{icon}</div> : null}
      </div>
      <p className={`font-display text-4xl font-bold leading-none ${accentClass[accent]}`}>{value}</p>
      {helper ? <p className="mt-4 text-sm text-charcoal/65">{helper}</p> : null}
    </motion.div>
  );
}
