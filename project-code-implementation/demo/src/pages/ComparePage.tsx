import type { EChartsOption } from "echarts";

import { EChartPanel } from "../components/EChartPanel";
import { MetricCard } from "../components/MetricCard";
import { useJsonData } from "../lib/data";
import { formatMetric } from "../lib/format";
import type { CompareSummary } from "../lib/types";

export function ComparePage() {
  const { data, loading, error } = useJsonData<CompareSummary>("compare.json");

  if (loading) return <div className="panel p-8 text-sm text-charcoal/60">Loading comparison…</div>;
  if (error || !data) return <div className="panel p-8 text-sm text-brick">{error ?? "Comparison load failed."}</div>;

  const ours = data.methods.find((method) => method.methodKey === data.primaryMethodKey) ?? data.methods[0];
  const bestModularity = [...data.methods].sort((a, b) => b.weightedModularity - a.weightedModularity)[0];
  const bestDensity = [...data.methods].sort((a, b) => b.weightedMeanDensity - a.weightedMeanDensity)[0];
  const bestClustering = [...data.methods].sort((a, b) => b.weightedMeanClustering - a.weightedMeanClustering)[0];

  return (
    <div className="space-y-6">
      <div>
        <p className="section-kicker">Grouping comparison</p>
        <h2 className="font-display text-4xl font-bold tracking-tight">Encoding-tree structure quality</h2>
        <p className="mt-2 max-w-3xl text-sm text-charcoal/65">
          This page compares grouping methods using unsupervised structural criteria. Purity is retained only as a
          label-aware external reference and is not treated as the primary optimization target.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard title="Our communities" value={`${ours.communities}`} accent="cobalt" />
        <MetricCard title="Our entropy" value={formatMetric(ours.structuralEntropy)} accent="acid" />
        <MetricCard title="Best modularity" value={formatMetric(bestModularity.weightedModularity)} helper={bestModularity.methodName} />
        <MetricCard title="Best density" value={formatMetric(bestDensity.weightedMeanDensity)} helper={bestDensity.methodName} accent="brick" />
        <MetricCard
          title="Best clustering"
          value={formatMetric(bestClustering.weightedMeanClustering)}
          helper={bestClustering.methodName}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="panel p-5">
          <p className="section-kicker">Structural metrics</p>
          <EChartPanel option={buildStructuralMetricOption(data)} className="h-[380px] w-full" />
        </div>
        <div className="panel p-5">
          <p className="section-kicker">Partition granularity</p>
          <EChartPanel option={buildGranularityOption(data)} className="h-[380px] w-full" />
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="panel p-5">
          <p className="section-kicker">Conductance and external purity</p>
          <EChartPanel option={buildPurityConductanceOption(data)} className="h-[380px] w-full" />
        </div>
        <div className="panel p-5">
          <p className="section-kicker">Community archetypes (ours)</p>
          <EChartPanel option={buildArchetypeOption(data)} className="h-[380px] w-full" />
        </div>
      </div>

      <div className="panel p-6">
        <p className="section-kicker">Representative communities</p>
        <div className="mt-5 grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {data.representativeCommunities.slice(0, 6).map((item) => (
            <div key={`${item.archetype}-${item.communityId}`} className="rounded-[1.2rem] border border-charcoal/10 bg-white/80 px-4 py-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-display text-xl font-bold">{item.communityId}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.18em] text-charcoal/45">{item.archetype}</p>
                </div>
                <div className="text-right text-xs text-charcoal/55">
                  <div>size {item.communitySize}</div>
                  <div>depth {formatMetric(item.encodingDepth, 1)}</div>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3 text-xs text-charcoal/65">
                <div>purity {formatMetric(item.purity, 4)}</div>
                <div>bot ratio {formatMetric(item.botRatio, 4)}</div>
                <div>density {formatMetric(item.density, 4)}</div>
                <div>clustering {formatMetric(item.clusteringCoefficient, 4)}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-5 text-sm leading-7 text-charcoal/65">{data.purityNote}</p>
      </div>
    </div>
  );
}

function buildStructuralMetricOption(data: CompareSummary): EChartsOption {
  const names = data.methods.map((item) => item.methodName);
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 8 },
    grid: { left: 56, right: 18, top: 54, bottom: 48 },
    xAxis: { type: "category" as const, data: names },
    yAxis: { type: "value" as const },
    series: [
      {
        name: "Structural entropy",
        type: "bar",
        data: data.methods.map((item) => item.structuralEntropy),
        itemStyle: { color: "#111111" },
      },
      {
        name: "Modularity",
        type: "bar",
        data: data.methods.map((item) => item.weightedModularity),
        itemStyle: { color: "#183b77" },
      },
      {
        name: "Mean density",
        type: "bar",
        data: data.methods.map((item) => item.weightedMeanDensity),
        itemStyle: { color: "#a8402d" },
      },
      {
        name: "Mean clustering",
        type: "bar",
        data: data.methods.map((item) => item.weightedMeanClustering),
        itemStyle: { color: "#6f8f14" },
      },
    ],
  };
}

function buildGranularityOption(data: CompareSummary): EChartsOption {
  const names = data.methods.map((item) => item.methodName);
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 8 },
    grid: { left: 56, right: 18, top: 54, bottom: 48 },
    xAxis: { type: "category" as const, data: names },
    yAxis: { type: "value" as const },
    series: [
      {
        name: "Communities",
        type: "bar",
        data: data.methods.map((item) => item.communities),
        itemStyle: { color: "#111111" },
      },
      {
        name: "Largest community",
        type: "bar",
        data: data.methods.map((item) => item.largestCommunity),
        itemStyle: { color: "#a8402d" },
      },
      {
        name: "Median community",
        type: "bar",
        data: data.methods.map((item) => item.medianCommunity),
        itemStyle: { color: "#183b77" },
      },
    ],
  };
}

function buildPurityConductanceOption(data: CompareSummary): EChartsOption {
  const names = data.methods.map((item) => item.methodName);
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 8 },
    grid: { left: 56, right: 18, top: 54, bottom: 48 },
    xAxis: { type: "category" as const, data: names },
    yAxis: { type: "value" as const, min: 0, max: 1 },
    series: [
      {
        name: "Mean conductance",
        type: "bar",
        data: data.methods.map((item) => item.weightedMeanConductance),
        itemStyle: { color: "#183b77" },
      },
      {
        name: "Global purity (external)",
        type: "bar",
        data: data.methods.map((item) => item.globalPurity ?? 0),
        itemStyle: { color: "#a8402d" },
      },
    ],
  };
}

function buildArchetypeOption(data: CompareSummary): EChartsOption {
  const entries = Object.entries(data.archetypeCounts);
  return {
    tooltip: { trigger: "item" },
    series: [
      {
        type: "pie",
        radius: ["36%", "70%"],
        center: ["50%", "52%"],
        label: {
          color: "#111111",
          formatter: "{b}\n{c}",
        },
        data: entries.map(([name, value], index) => ({
          name,
          value,
          itemStyle: {
            color: ["#111111", "#183b77", "#a8402d", "#6f8f14"][index % 4],
          },
        })),
      },
    ],
  };
}
