interface PipelineStripProps {
  steps: string[];
}

export function PipelineStrip({ steps }: PipelineStripProps) {
  return (
    <div className="panel-accent relative overflow-hidden p-6">
      <div
        className="absolute inset-0 bg-cover bg-center opacity-30"
        style={{ backgroundImage: `url(${import.meta.env.BASE_URL}visuals/hero-heatmap.png)` }}
      />
      <div className="relative">
        <p className="section-kicker text-paper/60">Current production pipeline</p>
        <div className="mt-4 grid gap-3 md:grid-cols-4 xl:grid-cols-6">
          {steps.map((step, index) => (
            <div key={step} className="rounded-[1.2rem] border border-white/12 bg-white/6 px-4 py-4">
              <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-paper/45">Step {index + 1}</p>
              <p className="mt-3 text-sm font-medium leading-6 text-paper">{step}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
