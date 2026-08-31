import { useMemo, useState } from "react";
import { createColumnHelper, flexRender, getCoreRowModel, getSortedRowModel, useReactTable, type SortingState } from "@tanstack/react-table";

import { CommunityInspector } from "../components/CommunityInspector";
import { UserDrawer } from "../components/UserDrawer";
import { useJsonData } from "../lib/data";
import { formatMetric } from "../lib/format";
import type { CommunityRecord, UserRecord } from "../lib/types";
import { useUiStore } from "../store/uiStore";

const columnHelper = createColumnHelper<CommunityRecord>();

const columns = [
  columnHelper.accessor("communityId", { header: "Community" }),
  columnHelper.accessor("communitySize", { header: "Size" }),
  columnHelper.accessor("purity", {
    header: "Purity",
    cell: (info) => formatMetric(info.getValue()),
  }),
  columnHelper.accessor("density", {
    header: "Density",
    cell: (info) => formatMetric(info.getValue()),
  }),
  columnHelper.accessor("clusteringCoefficient", {
    header: "Clustering",
    cell: (info) => formatMetric(info.getValue()),
  }),
  columnHelper.accessor("averageDegree", {
    header: "Avg degree",
    cell: (info) => formatMetric(info.getValue()),
  }),
  columnHelper.accessor("encodingDepth", {
    header: "Depth",
    cell: (info) => formatMetric(info.getValue() ?? 0, 1),
  }),
  columnHelper.accessor("archetype", { header: "Archetype" }),
];

export function CommunitiesPage() {
  const { data: communitiesData, loading, error } = useJsonData<{ communities: CommunityRecord[] }>("communities.json");
  const { data: graphData } = useJsonData<{ subgraphs: Record<string, { userIds: string[] }> }>("graph.json");
  const { data: usersData } = useJsonData<{ users: UserRecord[] }>("users.json");
  const [sorting, setSorting] = useState<SortingState>([{ id: "communitySize", desc: true }]);

  const selectedCommunityId = useUiStore((state) => state.selectedCommunityId);
  const setSelectedCommunityId = useUiStore((state) => state.setSelectedCommunityId);
  const selectedUserId = useUiStore((state) => state.selectedUserId);
  const setSelectedUserId = useUiStore((state) => state.setSelectedUserId);

  const communities = communitiesData?.communities ?? [];
  const table = useReactTable({
    data: communities,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const communitiesById = useMemo(() => new Map(communities.map((item) => [item.communityId, item])), [communities]);
  const usersById = useMemo(() => new Map((usersData?.users ?? []).map((item) => [item.userId, item])), [usersData]);
  const selectedCommunity = selectedCommunityId ? communitiesById.get(selectedCommunityId) ?? null : null;
  const selectedCommunityUsers = useMemo(() => {
    if (!selectedCommunity || !graphData) return [];
    return (graphData.subgraphs[selectedCommunity.communityId]?.userIds ?? [])
      .map((id) => usersById.get(id))
      .filter((item): item is UserRecord => Boolean(item))
      .sort((a, b) => b.followersCount - a.followersCount || b.tweetsTotal - a.tweetsTotal || a.userId.localeCompare(b.userId));
  }, [graphData, selectedCommunity, usersById]);

  if (loading) return <div className="panel p-8 text-sm text-charcoal/60">Loading communities…</div>;
  if (error) return <div className="panel p-8 text-sm text-brick">{error}</div>;

  return (
    <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
      <div className="panel overflow-hidden p-4">
        <div className="mb-4 px-2">
          <p className="section-kicker">Community table</p>
          <h2 className="font-display text-4xl font-bold tracking-tight">Community mix and structure</h2>
        </div>
        <div className="overflow-auto">
          <table className="data-table min-w-full">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <th
                      key={header.id}
                      onClick={header.column.getToggleSortingHandler()}
                      className="cursor-pointer select-none"
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className={`cursor-pointer transition hover:bg-charcoal/4 ${
                    row.original.communityId === selectedCommunityId ? "bg-charcoal/5" : ""
                  }`}
                  onClick={() => {
                    setSelectedCommunityId(row.original.communityId);
                    setSelectedUserId(null);
                  }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="xl:sticky xl:top-28 xl:self-start">
        <CommunityInspector
          community={selectedCommunity}
          users={selectedCommunityUsers}
          onSelectUser={(userId) => setSelectedUserId(userId)}
        />
      </div>
      <UserDrawer user={selectedUserId ? usersById.get(selectedUserId) ?? null : null} onClose={() => setSelectedUserId(null)} />
    </div>
  );
}
