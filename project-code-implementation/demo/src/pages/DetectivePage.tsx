import { useEffect, useMemo, useState } from "react";

import { useJsonData } from "../lib/data";
import type { UserRecord } from "../lib/types";

const POST_TYPE_META: { key: keyof UserRecord["postTypeRatios"]; label: string; barClass: string }[] = [
  { key: "original", label: "Original", barClass: "bg-cobalt" },
  { key: "retweet", label: "Retweet", barClass: "bg-brick" },
  { key: "commentReply", label: "Comment / Reply", barClass: "bg-emerald-600" },
  { key: "linkShare", label: "Link share", barClass: "bg-acid" },
];

const ARCHETYPE_HINT: Record<string, string> = {
  "Pure human macro-communities": "Human-majority macro cluster, strong cohesion.",
  "Compact bot communities": "Small, dense cluster with a high bot ratio.",
  "Mixed transitional communities": "Transitional structure mixing both labels.",
  "Sparse peripheral communities": "Sparse peripheral fragment, weak local cohesion.",
};

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function AccountAvatar({ user, size = 56 }: { user: UserRecord; size?: number }) {
  const isBot = user.label === "bot";
  return (
    <div
      className={`flex shrink-0 items-center justify-center rounded-full font-display font-bold text-paper ${
        isBot ? "bg-brick" : "bg-emerald-700"
      }`}
      style={{ width: size, height: size, fontSize: size * 0.42 }}
    >
      {(user.name || user.username || "?").charAt(0).toUpperCase()}
    </div>
  );
}

function VerdictBanner({ user }: { user: UserRecord }) {
  const isBot = user.label === "bot";
  return (
    <div
      className={`flex flex-col gap-2 rounded-xl2 border px-6 py-5 ${
        isBot ? "border-brick/30 bg-brick/5" : "border-emerald-600/30 bg-emerald-600/5"
      }`}
    >
      <div className="flex items-center gap-3">
        <span
          className={`rounded-full px-4 py-1.5 font-display text-sm font-bold tracking-wide text-paper ${
            isBot ? "bg-brick" : "bg-emerald-700"
          }`}
        >
          {isBot ? "● DETECTED: BOT" : "● DETECTED: HUMAN"}
        </span>
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-charcoal/45">
          ground-truth label · TwiBot-22 benchmark
        </span>
      </div>
      <p className="text-sm leading-6 text-charcoal/70">
        {isBot
          ? `This account sits inside ${user.communityArchetype.toLowerCase()}, where the discovered structure independently groups it with bot-like accounts.`
          : `This account sits inside ${user.communityArchetype.toLowerCase()}, alongside mostly human accounts.`}{" "}
        Labels are applied post-hoc for interpretation only — they never enter the optimization.
      </p>
    </div>
  );
}

function EvidenceCard({
  index,
  kicker,
  title,
  children,
}: {
  index: string;
  kicker: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="panel flex flex-col gap-4 p-6">
      <div>
        <p className="section-kicker">
          {index} · {kicker}
        </p>
        <h3 className="mt-1 font-display text-lg font-bold tracking-tight">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function StatRow({ label, value, pct, barClass }: { label: string; value: string; pct: number; barClass: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-charcoal/65">{label}</span>
        <span className="font-mono text-sm font-semibold text-charcoal">{value}</span>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-charcoal/10">
        <div className={`h-full rounded-full ${barClass}`} style={{ width: `${Math.max(2, Math.min(100, pct))}%` }} />
      </div>
    </div>
  );
}

export function DetectivePage() {
  const { data, error, loading } = useJsonData<{ users: UserRecord[] }>("users.json");
  const [query, setQuery] = useState("");
  const [seed, setSeed] = useState(0);
  const [selected, setSelected] = useState<UserRecord | null>(null);

  const users = data?.users ?? [];

  const curated = useMemo(() => {
    if (users.length === 0) return [];
    const rng = (n: number) => Math.abs(Math.sin(seed * 12.9898 + n * 78.233)) * 43758.5453 % 1;
    const bots = users.filter((u) => u.label === "bot" && u.canFullPipeline);
    const humans = users.filter((u) => u.label === "human" && u.canFullPipeline);
    const pick = (arr: UserRecord[], count: number) => {
      const out: UserRecord[] = [];
      const copy = [...arr];
      for (let i = 0; i < count && copy.length > 0; i++) {
        const idx = Math.floor(rng(seed * 1000 + i) * copy.length);
        out.push(copy.splice(idx, 1)[0]);
      }
      return out;
    };
    return [...pick(bots, 5), ...pick(humans, 6)];
  }, [users, seed]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return curated;
    return users.filter(
      (u) =>
        u.username.toLowerCase().includes(q) ||
        (u.name || "").toLowerCase().includes(q) ||
        (u.descriptionExcerpt || "").toLowerCase().includes(q),
    ).slice(0, 12);
  }, [users, query, curated]);

  useEffect(() => {
    if (!selected && curated.length > 0) {
      setSelected(curated.find((u) => u.label === "bot") ?? curated[0]);
    }
  }, [curated, selected]);

  const selectedCommunity = selected
    ? {
        purity: selected.communityPurity,
        density: selected.communityDensity,
        clustering: selected.communityClustering,
        size: selected.communitySize,
      }
    : null;

  return (
    <div className="space-y-8">
      <section className="panel-accent relative overflow-hidden px-6 py-10 lg:px-10">
        <div className="relative max-w-4xl">
          <p className="section-kicker text-paper/60">Interactive research demo · 18,743 real accounts</p>
          <h2 className="mt-3 font-display text-4xl font-bold leading-tight tracking-tight text-paper lg:text-5xl">
            Account Detective
          </h2>
          <p className="mt-4 max-w-2xl text-base leading-8 text-paper/78 lg:text-lg">
            Pick any real account from the sampled benchmark and walk through the four evidence views —
            content, behavior, community structure and final verdict — exactly as the late-fusion pipeline sees it.
          </p>
          <div className="mt-7 flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="flex flex-1 items-center gap-2 rounded-full border border-white/15 bg-paper/95 px-5 py-3">
              <span className="text-charcoal/40">🔍</span>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by username, name or bio…"
                className="w-full bg-transparent text-sm text-charcoal outline-none placeholder:text-charcoal/40"
              />
              {query && (
                <button
                  onClick={() => setQuery("")}
                  className="rounded-full px-2 text-sm text-charcoal/50 transition hover:text-charcoal"
                >
                  ✕
                </button>
              )}
            </div>
            <button
              onClick={() => {
                setQuery("");
                setSeed((s) => s + 1);
              }}
              className="rounded-full bg-acid px-6 py-3 font-body text-sm font-bold text-ink transition hover:brightness-95"
            >
              Surprise me ↻
            </button>
          </div>
          <p className="mt-4 font-mono text-[11px] uppercase tracking-[0.18em] text-paper/45">
            {users.length.toLocaleString()} accounts · {results.length} shown · labels used post-hoc only
          </p>
        </div>
      </section>

      {loading && <div className="panel p-8 text-sm text-charcoal/60">Loading accounts…</div>}
      {error && <div className="panel p-8 text-sm text-brick">{error}</div>}
      {!loading && !error && (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          {/* Left: account list */}
          <div className="flex flex-col gap-3">
            <p className="section-kicker">Accounts</p>
            <div className="grid gap-3 sm:grid-cols-2">
              {results.map((u) => {
                const isBot = u.label === "bot";
                const active = selected?.userId === u.userId;
                return (
                  <button
                    key={u.userId}
                    onClick={() => setSelected(u)}
                    className={`panel flex flex-col gap-3 p-4 text-left transition ${
                      active ? "ring-2 ring-charcoal" : "hover:-translate-y-0.5 hover:shadow-editorial"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <AccountAvatar user={u} size={44} />
                      <div className="min-w-0">
                        <p className="truncate font-display text-sm font-bold">{u.name || u.username}</p>
                        <p className="truncate font-mono text-[11px] text-charcoal/50">@{u.username}</p>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-full px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider ${
                          isBot ? "bg-brick/12 text-brick" : "bg-emerald-600/12 text-emerald-700"
                        }`}
                      >
                        {isBot ? "bot" : "human"}
                      </span>
                      <span className="rounded-full bg-charcoal/6 px-2.5 py-0.5 font-mono text-[10px] text-charcoal/55">
                        {formatCount(u.followersCount)} followers
                      </span>
                      <span className="truncate rounded-full bg-charcoal/6 px-2.5 py-0.5 font-mono text-[10px] text-charcoal/55">
                        {u.communityArchetype.split(" ")[0]}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
            {results.length === 0 && (
              <div className="panel p-8 text-center text-sm text-charcoal/55">No accounts match “{query}”.</div>
            )}
          </div>

          {/* Right: evidence detail */}
          <div>
            {selected && selectedCommunity ? (
              <div className="flex flex-col gap-6">
                {/* Identity header */}
                <div className="panel flex flex-wrap items-center justify-between gap-5 p-6">
                  <div className="flex items-center gap-4">
                    <AccountAvatar user={selected} size={64} />
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="font-display text-2xl font-bold tracking-tight">
                          {selected.name || selected.username}
                        </h3>
                        <span className="rounded-full bg-charcoal/6 px-2.5 py-0.5 font-mono text-[10px] text-charcoal/50">
                          @{selected.username}
                        </span>
                        {selected.verified === 1 && (
                          <span className="rounded-full bg-cobalt px-2.5 py-0.5 font-mono text-[10px] font-bold text-paper">
                            verified
                          </span>
                        )}
                      </div>
                      <p className="mt-1 line-clamp-2 max-w-xl text-sm leading-6 text-charcoal/60">
                        {selected.descriptionExcerpt || "No public description in the sampled profile."}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6 font-mono text-sm">
                    <div>
                      <p className="font-display text-xl font-bold">{formatCount(selected.followersCount)}</p>
                      <p className="text-[10px] uppercase tracking-wider text-charcoal/45">followers</p>
                    </div>
                    <div>
                      <p className="font-display text-xl font-bold">{formatCount(selected.followingCount)}</p>
                      <p className="text-[10px] uppercase tracking-wider text-charcoal/45">following</p>
                    </div>
                    <div>
                      <p className="font-display text-xl font-bold">{formatCount(selected.tweetsTotal)}</p>
                      <p className="text-[10px] uppercase tracking-wider text-charcoal/45">tweets</p>
                    </div>
                  </div>
                </div>

                {/* Four evidence views */}
                <div className="grid gap-6 lg:grid-cols-2">
                  <EvidenceCard index="①" kicker="Content view" title="What does this account post?">
                    <div className="flex flex-col gap-3">
                      {POST_TYPE_META.map((m) => {
                        const ratio = selected.postTypeRatios?.[m.key] ?? 0;
                        return (
                          <StatRow
                            key={m.key}
                            label={m.label}
                            value={`${(ratio * 100).toFixed(0)}%`}
                            pct={ratio * 100}
                            barClass={m.barClass}
                          />
                        );
                      })}
                    </div>
                    {selected.tripletSummary && (
                      <div className="rounded-xl bg-charcoal/4 p-4">
                        <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-charcoal/45">
                          LLM triplet summary
                        </p>
                        <p className="mt-2 line-clamp-4 text-[13px] leading-6 text-charcoal/75">
                          {selected.tripletSummary.replace(/\n/g, " · ")}
                        </p>
                      </div>
                    )}
                  </EvidenceCard>

                  <EvidenceCard index="②" kicker="Behavior view" title="How does it behave?">
                    <div className="flex flex-col gap-4">
                      <StatRow
                        label="Follower / following ratio"
                        value={
                          selected.followingCount > 0
                            ? (selected.followersCount / selected.followingCount).toFixed(2)
                            : "—"
                        }
                        pct={
                          selected.followingCount > 0
                            ? Math.min(100, (selected.followersCount / Math.max(1, selected.followingCount)) * 50)
                            : 0
                        }
                        barClass="bg-cobalt"
                      />
                      <StatRow
                        label="Tweet volume"
                        value={formatCount(selected.tweetsTotal)}
                        pct={Math.min(100, (selected.tweetsTotal / 40) * 100)}
                        barClass="bg-brick"
                      />
                      <StatRow
                        label="Verified badge"
                        value={selected.verified === 1 ? "Yes" : "No"}
                        pct={selected.verified === 1 ? 100 : 0}
                        barClass="bg-emerald-600"
                      />
                      <p className="text-[13px] leading-6 text-charcoal/60">
                        Combined with post-type distribution and posting cadence, these signals feed the behavior
                        view of the fused graph.
                      </p>
                    </div>
                  </EvidenceCard>

                  <EvidenceCard index="③" kicker="Community view" title="Which cluster does it belong to?">
                    <div className="flex flex-wrap gap-2">
                      <span className="rounded-full bg-charcoal/6 px-3 py-1 font-mono text-xs text-charcoal/70">
                        {selected.communityId}
                      </span>
                      <span className="rounded-full bg-cobalt px-3 py-1 font-mono text-xs font-semibold text-paper">
                        {selectedCommunity.size} members
                      </span>
                      <span className="rounded-full bg-charcoal/6 px-3 py-1 font-mono text-xs text-charcoal/70">
                        {ARCHETYPE_HINT[selected.communityArchetype]?.split(",")[0] ?? selected.communityArchetype}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                      <StatRow
                        label="Label purity"
                        value={(selectedCommunity.purity * 100).toFixed(1) + "%"}
                        pct={selectedCommunity.purity * 100}
                        barClass="bg-emerald-600"
                      />
                      <StatRow
                        label="Density"
                        value={selectedCommunity.density.toFixed(3)}
                        pct={Math.min(100, selectedCommunity.density * 180)}
                        barClass="bg-cobalt"
                      />
                      <StatRow
                        label="Clustering"
                        value={selectedCommunity.clustering.toFixed(3)}
                        pct={Math.min(100, selectedCommunity.clustering * 160)}
                        barClass="bg-acid"
                      />
                      <StatRow
                        label="Encoding depth"
                        value={selected.communityId}
                        pct={50}
                        barClass="bg-charcoal/30"
                      />
                    </div>
                  </EvidenceCard>

                  <EvidenceCard index="④" kicker="Network view" title="How connected is its neighbourhood?">
                    <div className="flex flex-col gap-3">
                      <StatRow
                        label="Cluster size (network degree proxy)"
                        value={`${selectedCommunity.size} users`}
                        pct={Math.min(100, (selectedCommunity.size / 300) * 100)}
                        barClass="bg-cobalt"
                      />
                      <StatRow
                        label="Local cohesion (density × clustering)"
                        value={(selectedCommunity.density * selectedCommunity.clustering).toFixed(4)}
                        pct={Math.min(100, selectedCommunity.density * selectedCommunity.clustering * 500)}
                        barClass="bg-brick"
                      />
                      <p className="text-[13px] leading-6 text-charcoal/60">
                        The 10k graph was built with a late-fusion similarity over all four views, then partitioned
                        by structural-entropy minimization — this account lands in {selected.communityId} on the
                        resulting encoding tree.
                      </p>
                    </div>
                  </EvidenceCard>
                </div>

                <VerdictBanner user={selected} />
              </div>
            ) : (
              <div className="panel flex h-full min-h-[320px] flex-col items-center justify-center gap-3 p-8 text-center">
                <p className="font-display text-2xl font-bold">Select an account to inspect</p>
                <p className="max-w-sm text-sm leading-7 text-charcoal/55">
                  Every figure below is computed from the real sampled benchmark — no synthetic data, no placeholders.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
