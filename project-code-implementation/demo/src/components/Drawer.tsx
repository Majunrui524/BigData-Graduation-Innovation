import { AnimatePresence, motion } from "framer-motion";
import type { PropsWithChildren } from "react";

interface DrawerProps extends PropsWithChildren {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
}

export function Drawer({ open, onClose, title, subtitle, children }: DrawerProps) {
  return (
    <AnimatePresence>
      {open ? (
        <>
          <motion.button
            type="button"
            className="fixed inset-0 z-40 bg-charcoal/25 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            className="fixed right-0 top-0 z-50 h-full w-full max-w-[520px] overflow-y-auto border-l border-charcoal/10 bg-paper px-6 py-6 shadow-[0_0_80px_rgba(17,17,17,0.18)]"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ duration: 0.28, ease: "easeOut" }}
          >
            <div className="mb-6 flex items-start justify-between gap-4">
              <div>
                <p className="section-kicker">Detail drawer</p>
                <h3 className="font-display text-3xl font-bold tracking-tight">{title}</h3>
                {subtitle ? <p className="mt-2 text-sm text-charcoal/65">{subtitle}</p> : null}
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-full border border-charcoal/10 px-3 py-2 font-mono text-xs uppercase tracking-[0.18em] text-charcoal/65 hover:border-charcoal/30 hover:text-charcoal"
              >
                Close
              </button>
            </div>
            {children}
          </motion.aside>
        </>
      ) : null}
    </AnimatePresence>
  );
}
