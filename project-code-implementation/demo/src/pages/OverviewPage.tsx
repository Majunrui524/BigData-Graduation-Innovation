import { Link } from "react-router-dom";
import { MetricCard } from "../components/MetricCard";
import { PipelineStrip } from "../components/PipelineStrip";
import { useJsonData } from "../lib/data";
import { formatMetric, formatNumber } from "../lib/format";
import type { OverviewSummary } from "../lib/types";

export function OverviewPage() {
  const { data, error, loading } = useJsonData<OverviewSummary>("overview.json");

  if (loading) return <LoadingState />;
  if (error || !data) return <ErrorState message={error ?? "Failed to load overview data."} />;

  return (
    <div className="space-y-8">
      <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="panel-accent relative overflow-hidden px-6 py-8 lg:px-8 lg:py-10">
          <div
            className="absolute inset-0 bg-cover bg-center opacity-30"
            style={{ backgroundImage: `url(${import.meta.env.BASE_URL}visuals/hero-heatmap.png)` }}
          />
          <div className="relative max-w-3xl">
            <p className="section-kicker text-paper/60">Research presentation interface</p>
            <h2 className="mt-3 font-display text-4xl font-bold leading-tight tracking-tight text-paper lg:text-6xl">
              {data.title}
            </h2>
            <p className="mt-5 max-w-2xl text-base leading-8 text-paper/78 lg:text-lg">{data.subtitle}</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <span className="tag border-white/15 bg-white/10 text-paper/75">18,743 users</span>
              <span className="tag border-white/15 bg-white/10 text-paper/75">late fusion</span>
              <span className="tag border-white/15 bg-white/10 text-paper/75">structural entropy</span>
              <span className="tag border-white/15 bg-white/10 text-paper/75">encoding tree</span>
            </div>
            <div className="mt-7 flex flex-wrap items-center gap-4">
              <Link
                to="/detective"
                className="rounded-full bg-acid px-7 py-3.5 font-body text-sm font-bold text-ink transition hover:brightness-95"
              >
                🕵️ Try the Account Detective →
              </Link>
              <Link
                to="/graph"
                className="rounded-full border border-white/20 px-7 py-3.5 font-body text-sm font-semibold text-paper/85 transition hover:bg-white/10 hover:text-paper"
              >
                Explore the community graph
              </Link>
            </div>
          </div>
        </div>
        <div
          className="panel relative overflow-hidden p-6"
          style={{
            backgroundImage: `url(${import.meta.env.BASE_URL}visuals/community-risk-poster.png)`,
            backgroundSize: "cover",
          }}
        >
          <div className="max-w-sm rounded-[1.4rem] border border-charcoal/10 bg-paper/88 p-5 backdrop-blur">
            <p className="section-kicker">Executive summary</p>
            <div className="mt-4 space-y-3 text-sm leading-7 text-charcoal/75">
              {data.takeaways.map((item) => (
                <p key={item}>{item}</p>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Users"
          value={formatNumber(data.sample.users)}
          helper={`${formatNumber(data.sample.humans)} humans · ${formatNumber(data.sample.bots)} bots`}
          accent="brick"
        />
        <MetricCard
          title="Graph edges"
          value={formatNumber(data.graph.undirectedEdges)}
          helper={`k=${data.graph.k} · candidate_k=${data.graph.candidateK}`}
          accent="cobalt"
        />
        <MetricCard
          title="Communities"
          value={formatNumber(data.graph.communities)}
          helper={`largest ${formatNumber(data.graph.largestCommunity)} · median ${formatMetric(data.graph.medianCommunity, 1)}`}
          accent="acid"
        />
        <MetricCard
          title="Entropy drop"
          value={formatMetric(data.graph.initialEntropy - data.graph.finalEntropy)}
          helper={`${formatMetric(data.graph.initialEntropy)} → ${formatMetric(data.graph.finalEntropy)}`}
        />
      </section>

      <PipelineStrip steps={data.pipeline} />

      <section className="grid gap-6 xl:grid-cols-2">
        <div className="panel p-6">
          <p className="section-kicker">Primary structural quality</p>
          <h3 className="mt-2 font-display text-3xl font-bold">Encoding-tree partition quality</h3>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            <MetricCard title="Structural entropy" value={formatMetric(data.graph.finalEntropy)} />
            <MetricCard title="Modularity" value={formatMetric(data.graph.weightedModularity)} accent="brick" />
            <MetricCard title="Mean density" value={formatMetric(data.graph.weightedMeanDensity)} accent="cobalt" />
            <MetricCard title="Mean clustering" value={formatMetric(data.graph.weightedMeanClustering)} accent="acid" />
            <MetricCard title="Mean conductance" value={formatMetric(data.graph.weightedMeanConductance)} />
            <MetricCard title="External purity" value={formatMetric(data.graph.globalPurity)} />
          </div>
        </div>
        <div className="panel p-6">
          <p className="section-kicker">Grouping baselines</p>
          <h3 className="mt-2 font-display text-3xl font-bold">Comparable community methods</h3>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {data.groupingMethods.map((method) => (
              <MetricCard
                key={method.methodKey}
                title={method.methodName}
                value={formatMetric(method.structuralEntropy)}
                helper={`Q ${formatMetric(method.weightedModularity)} · D ${formatMetric(method.weightedMeanDensity)} · C ${formatMetric(method.weightedMeanClustering)}`}
                accent={method.methodKey === "structural_entropy" ? "acid" : "charcoal"}
              />
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <CommunityHighlights title="Pure human macro-communities" items={data.topPureHumanCommunities} />
        <CommunityHighlights title="Compact bot communities" items={data.topCompactBotCommunities} />
      </section>
    </div>
  );
}

function CommunityHighlights({
  title,
  items,
}: {
  title: string;
  items: OverviewSummary["topPureHumanCommunities"];
}) {
  return (
    <div className="panel p-6">
      <p className="section-kicker">{title}</p>
      <div className="mt-5 space-y-3">
        {items.map((item) => (
          <div key={item.communityId} className="rounded-[1.2rem] border border-charcoal/10 bg-white/80 px-4 py-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="font-display text-xl font-bold">{item.communityId}</p>
                <p className="mt-1 text-sm text-charcoal/65">
                  size {item.communitySize} · {item.archetype}
                </p>
              </div>
              <div className="text-right">
                <p className="font-mono text-sm uppercase tracking-[0.18em] text-charcoal/45">purity</p>
                <p className="mt-1 text-lg font-semibold">{formatMetric(item.purity, 4)}</p>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3 text-xs text-charcoal/60">
              <div>density {formatMetric(item.density, 4)}</div>
              <div>clustering {formatMetric(item.clusteringCoefficient, 4)}</div>
              <div>bot ratio {formatMetric(item.botRatio, 4)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LoadingState() {
  return <div className="panel p-8 text-sm text-charcoal/60">Loading overview…</div>;
}

function ErrorState({ message }: { message: string }) {
  return <div className="panel p-8 text-sm text-brick">{message}</div>;
}
