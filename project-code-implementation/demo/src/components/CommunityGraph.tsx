import Graph from "graphology";
import Sigma from "sigma";
import { useEffect, useMemo, useRef } from "react";

import { scoreToColor } from "../lib/format";
import type { CommunityEdge, CommunityNode, ScoreMode } from "../lib/types";

interface CommunityGraphProps {
  nodes: CommunityNode[];
  edges: CommunityEdge[];
  scoreMode: ScoreMode;
  onSelectCommunity: (communityId: string) => void;
  highlightedCommunityId?: string | null;
}

export function CommunityGraph({
  nodes,
  edges,
  scoreMode,
  onSelectCommunity,
  highlightedCommunityId,
}: CommunityGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const graphData = useMemo(() => {
    const graph = new Graph();
    nodes.forEach((node) => {
      const score = scoreMode === "density" ? node.density : node.clusteringCoefficient;
      graph.addNode(node.id, {
        x: node.x,
        y: node.y,
        label: node.id,
        size: Math.max(4, Math.sqrt(node.communitySize) * 1.3),
        color: scoreToColor(score),
        communitySize: node.communitySize,
      });
    });
    edges.forEach((edge) => {
      if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
      graph.addEdge(edge.source, edge.target, {
        size: Math.max(0.5, edge.weight * 2.4),
        color: "rgba(17,17,17,0.12)",
      });
    });
    return graph;
  }, [edges, nodes, scoreMode]);

  useEffect(() => {
    if (!containerRef.current) return;
    const renderer = new Sigma(graphData, containerRef.current, {
      renderEdgeLabels: false,
      labelDensity: 0.04,
      labelGridCellSize: 100,
      labelRenderedSizeThreshold: 12,
      zIndex: true,
      labelColor: { color: "#111111" },
      labelSize: 12,
      labelFont: "IBM Plex Sans",
    });

    renderer.on("clickNode", ({ node }) => onSelectCommunity(String(node)));

    if (highlightedCommunityId && graphData.hasNode(highlightedCommunityId)) {
      graphData.forEachNode((nodeId, attrs) => {
        graphData.setNodeAttribute(
          nodeId,
          "color",
          nodeId === highlightedCommunityId ? attrs.color : "rgba(17,17,17,0.18)",
        );
      });
    }

    return () => {
      renderer.kill();
    };
  }, [graphData, highlightedCommunityId, onSelectCommunity]);

  return <div ref={containerRef} className="h-[760px] w-full rounded-[1.4rem] bg-white/70" />;
}
