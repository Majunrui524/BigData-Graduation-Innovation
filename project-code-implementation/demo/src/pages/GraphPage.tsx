import { useMemo } from "react";

import { CommunityGraph } from "../components/CommunityGraph";
import { CommunityInspector } from "../components/CommunityInspector";
import { ScoreModeToggle } from "../components/ScoreModeToggle";
import { UserDrawer } from "../components/UserDrawer";
import { useJsonData } from "../lib/data";
import type { CommunityRecord, GraphBundle, UserRecord } from "../lib/types";
import { useUiStore } from "../store/uiStore";

export function GraphPage() {
  const { data: graphData, loading, error } = useJsonData<GraphBundle>("graph.json");
  const { data: communitiesData } = useJsonData<{ communities: CommunityRecord[] }>("communities.json");
  const { data: usersData } = useJsonData<{ users: UserRecord[] }>("users.json");

  const scoreMode = useUiStore((state) => state.scoreMode);
  const selectedCommunityId = useUiStore((state) => state.selectedCommunityId);
  const setSelectedCommunityId = useUiStore((state) => state.setSelectedCommunityId);
  const selectedUserId = useUiStore((state) => state.selectedUserId);
  const setSelectedUserId = useUiStore((state) => state.setSelectedUserId);

  const communitiesById = useMemo(
    () => new Map((communitiesData?.communities ?? []).map((item) => [item.communityId, item])),
    [communitiesData],
  );
  const usersById = useMemo(() => new Map((usersData?.users ?? []).map((item) => [item.userId, item])), [usersData]);

  const selectedCommunity = selectedCommunityId ? communitiesById.get(selectedCommunityId) ?? null : null;
  const selectedCommunityUsers = useMemo(() => {
    if (!selectedCommunity || !graphData) return [];
    const userIds = graphData.subgraphs[selectedCommunity.communityId]?.userIds ?? [];
    return userIds
      .map((id) => usersById.get(id))
      .filter((item): item is UserRecord => Boolean(item))
      .sort((a, b) => b.followersCount - a.followersCount || b.tweetsTotal - a.tweetsTotal || a.userId.localeCompare(b.userId));
  }, [graphData, selectedCommunity, usersById]);
  const selectedUser = selectedUserId ? usersById.get(selectedUserId) ?? null : null;

  if (loading) return <div className="panel p-8 text-sm text-charcoal/60">Loading graph…</div>;
  if (error || !graphData) return <div className="panel p-8 text-sm text-brick">{error ?? "Graph load failed."}</div>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-kicker">Community-first graph</p>
          <h2 className="font-display text-4xl font-bold tracking-tight">Structural entropy community graph</h2>
          <p className="mt-2 text-sm text-charcoal/65">
            Default view aggregates the 18,743-user graph into community nodes for stable exploration. Use the toggle on
            the right to switch node coloring between internal density and local clustering coefficient.
          </p>
        </div>
        <ScoreModeToggle />
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.4fr_0.8fr]">
        <div className="panel p-4">
          <CommunityGraph
            nodes={graphData.nodes}
            edges={graphData.edges}
            scoreMode={scoreMode}
            highlightedCommunityId={selectedCommunityId}
            onSelectCommunity={(communityId) => {
              setSelectedCommunityId(communityId);
              setSelectedUserId(null);
            }}
          />
        </div>
        <CommunityInspector
          community={selectedCommunity}
          users={selectedCommunityUsers}
          onSelectUser={(userId) => setSelectedUserId(userId)}
        />
      </div>
      <UserDrawer user={selectedUser} onClose={() => setSelectedUserId(null)} />
    </div>
  );
}
