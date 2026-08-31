import { useUiStore } from "../store/uiStore";

export function ScoreModeToggle() {
  const scoreMode = useUiStore((state) => state.scoreMode);
  const setScoreMode = useUiStore((state) => state.setScoreMode);

  return (
    <div className="inline-flex rounded-full border border-charcoal/10 bg-white/75 p-1">
      {(["density", "clustering"] as const).map((mode) => {
        const active = scoreMode === mode;
        return (
          <button
            key={mode}
            type="button"
            onClick={() => setScoreMode(mode)}
            className={`rounded-full px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] transition ${
              active ? "bg-charcoal text-paper" : "text-charcoal/60 hover:text-charcoal"
            }`}
          >
            {mode === "density" ? "Density" : "Clustering"}
          </button>
        );
      })}
    </div>
  );
}
