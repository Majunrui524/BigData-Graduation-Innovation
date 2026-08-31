import { useMemo, useState } from "react";

import { UserDrawer } from "../components/UserDrawer";
import { useJsonData } from "../lib/data";
import { formatMetric } from "../lib/format";
import type { ErrorCase, ErrorsSummary, UserRecord } from "../lib/types";
import { useUiStore } from "../store/uiStore";

type ErrorTab = "fixed" | "regressed" | "unchanged";

export function ErrorsPage() {
  const { data, loading, error } = useJsonData<ErrorsSummary>("errors.json");
  const { data: usersData } = useJsonData<{ users: UserRecord[] }>("users.json");
  const [tab, setTab] = useState<ErrorTab>("fixed");
  const [query, setQuery] = useState("");
  const selectedUserId = useUiStore((state) => state.selectedUserId);
  const setSelectedUserId = useUiStore((state) => state.setSelectedUserId);

  const usersById = useMemo(() => new Map((usersData?.users ?? []).map((row) => [row.userId, row])), [usersData]);
  const selectedUser = selectedUserId ? usersById.get(selectedUserId) ?? null : null;

  if (loading) return <div className="panel p-8 text-sm text-charcoal/60">Loading error analysis…</div>;
  if (error || !data) return <div className="panel p-8 text-sm text-brick">{error ?? "Error analysis load failed."}</div>;

  const cases = tab === "fixed" ? data.fixedCases : tab === "regressed" ? data.regressedCases : data.unchangedErrors;
  const normalizedQuery = query.trim().toLowerCase();
  const filteredCases = cases.filter((item) => {
    if (!normalizedQuery) return true;
    const queryLooksLikeUserId = normalizedQuery.startsWith("u");
    const username = item.username.toLowerCase();
    const name = item.name.toLowerCase();
    const communityId = item.communityId.toLowerCase();
    const userId = item.userId.toLowerCase();
    return (
      username.includes(normalizedQuery) ||
      name.includes(normalizedQuery) ||
      communityId.includes(normalizedQuery) ||
      (queryLooksLikeUserId && userId.includes(normalizedQuery))
    );
  });
  const matchedCommunityIds = new Set(filteredCases.map((item) => item.communityId));
  const filteredCommunityChanges = data.communityChanges.filter((row) => {
    if (!normalizedQuery) return true;
    return row.communityId.toLowerCase().includes(normalizedQuery) || matchedCommunityIds.has(row.communityId);
  });

  return (
    <div className="space-y-6">
      <div>
        <p className="section-kicker">Failure modes</p>
        <h2 className="font-display text-4xl font-bold tracking-tight">Error analysis and community movement</h2>
      </div>

      <div className="panel p-5">
        <div className="flex flex-wrap items-center gap-3">
          {(["fixed", "regressed", "unchanged"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setTab(value)}
              className={`rounded-full border px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] ${
                tab === value ? "border-charcoal bg-charcoal text-paper" : "border-charcoal/10 bg-white/70 text-charcoal/60"
              }`}
            >
              {value}
            </button>
          ))}
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search username / name / community_id (prefix user_id with u)"
            className="ml-auto w-full max-w-sm rounded-full border border-charcoal/10 bg-paper px-4 py-3 text-sm outline-none ring-0 placeholder:text-charcoal/35 focus:border-charcoal/30"
          />
        </div>

        <div className="mt-5 grid gap-4 xl:grid-cols-[1.3fr_0.7fr]">
          <div className="overflow-auto rounded-[1.2rem] border border-charcoal/10">
            <table className="data-table min-w-full">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Community</th>
                  <th>Label</th>
                  <th>Baseline</th>
                  <th>Reranker</th>
                  <th>Δ score</th>
                </tr>
              </thead>
              <tbody>
              {filteredCases.slice(0, 250).map((item) => (
                  <tr key={`${tab}-${item.userId}`} className="cursor-pointer hover:bg-charcoal/4" onClick={() => setSelectedUserId(item.userId)}>
                    <td>{item.username || item.userId}</td>
                    <td>{item.communityId}</td>
                    <td className="uppercase">{item.label}</td>
                    <td>{item.baselinePredictedLabel}</td>
                    <td>{item.rerankerPredictedLabel}</td>
                    <td>{formatMetric(item.scoreDelta, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel bg-white/70 p-5 xl:sticky xl:top-28 xl:self-start xl:max-h-[calc(100vh-8rem)] xl:overflow-auto">
            <p className="section-kicker">Community net-gain summary</p>
            <div className="mt-4 space-y-3">
              {filteredCommunityChanges.slice(0, 8).map((row) => (
                <div key={row.communityId} className="rounded-[1.2rem] border border-charcoal/10 bg-paper px-4 py-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-display text-xl font-bold">{row.communityId}</p>
                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-charcoal/45">
                        size {row.communitySize}
                      </p>
                    </div>
                    <div className={`font-mono text-lg ${row.netGain >= 0 ? "text-[#6f8f14]" : "text-brick"}`}>
                      {row.netGain >= 0 ? "+" : ""}
                      {row.netGain}
                    </div>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-charcoal/60">
                    <div>fixed {row.fixedCount}</div>
                    <div>regressed {row.regressedCount}</div>
                    <div>base err {formatMetric(row.baselineErrorRate, 3)}</div>
                    <div>rerank err {formatMetric(row.rerankerErrorRate, 3)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <UserDrawer user={selectedUser} onClose={() => setSelectedUserId(null)} />
    </div>
  );
}
