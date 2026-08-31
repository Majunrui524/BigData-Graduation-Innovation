import type { CommunityRecord, UserRecord } from "../lib/types";
import { formatMetric } from "../lib/format";

interface CommunityInspectorProps {
  community: CommunityRecord | null;
  users: UserRecord[];
  onSelectUser: (userId: string) => void;
}

export function CommunityInspector({ community, users, onSelectUser }: CommunityInspectorProps) {
  if (!community) {
    return (
      <div className="panel flex h-full min-h-[420px] items-center justify-center p-8 text-center text-charcoal/55 xl:max-h-[calc(100vh-8rem)] xl:overflow-auto">
        Select a community to inspect its structure, archetype, and representative users.
      </div>
    );
  }

  return (
    <div className="panel h-full space-y-5 p-5 xl:max-h-[calc(100vh-8rem)] xl:overflow-auto">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="section-kicker">Community inspector</p>
          <h3 className="font-display text-3xl font-bold tracking-tight">{community.communityId}</h3>
          <p className="mt-2 text-sm text-charcoal/65">
            Encoding depth {community.encodingDepth ?? "N/A"} · Size {community.communitySize}
          </p>
        </div>
        <div className="tag">{community.archetype}</div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat title="Purity" value={formatMetric(community.purity)} />
        <Stat title="Density" value={formatMetric(community.density)} />
        <Stat title="Clustering" value={formatMetric(community.clusteringCoefficient)} />
        <Stat title="Avg degree" value={formatMetric(community.averageDegree)} />
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm">
        <Stat title="Train" value={`${community.trainCount}`} />
        <Stat title="Valid" value={`${community.validCount}`} />
        <Stat title="Test" value={`${community.testCount}`} />
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm">
        <Stat title="Human count" value={`${community.humanCount}`} />
        <Stat title="Bot count" value={`${community.botCount}`} />
        <Stat title="Bot ratio" value={formatMetric(community.botRatio)} />
      </div>

      <div className="rounded-[1.2rem] border border-charcoal/10 bg-paper p-4">
        <p className="section-kicker">Representative users</p>
        <div className="mt-3 space-y-3">
          {users.slice(0, 6).map((user) => (
            <button
              key={user.userId}
              type="button"
              onClick={() => onSelectUser(user.userId)}
              className="w-full rounded-2xl border border-charcoal/10 bg-white/80 px-4 py-4 text-left transition hover:-translate-y-0.5 hover:border-charcoal/30"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-semibold">{user.username || user.userId}</p>
                  <p className="mt-1 text-xs uppercase tracking-[0.18em] text-charcoal/45">{user.communityArchetype}</p>
                </div>
                <div className="text-right text-xs text-charcoal/55">
                  <div>{user.followersCount} followers</div>
                  <div>{user.tweetsTotal} tweets</div>
                </div>
              </div>
              <p className="mt-3 text-sm leading-6 text-charcoal/70">{user.descriptionExcerpt || "No profile description."}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-[1.2rem] border border-charcoal/10 bg-white/80 px-4 py-3">
      <p className="section-kicker">{title}</p>
      <p className="mt-2 text-lg font-semibold">{value}</p>
    </div>
  );
}
