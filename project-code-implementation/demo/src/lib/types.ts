export type ScoreMode = "density" | "clustering";

export interface OverviewSummary {
  title: string;
  subtitle: string;
  sample: {
    users: number;
    tweets: number;
    edges: number;
    humans: number;
    bots: number;
    train: number;
    valid: number;
    test: number;
  };
  pipeline: string[];
  graph: {
    users: number;
    undirectedEdges: number;
    communities: number;
    largestCommunity: number;
    medianCommunity: number;
    treeDepth: number;
    initialEntropy: number;
    finalEntropy: number;
    weightedModularity: number;
    weightedMeanDensity: number;
    weightedMeanClustering: number;
    weightedMeanConductance: number;
    globalPurity: number;
    channelCoverage: Record<string, number>;
    archetypeCounts: Record<string, number>;
    k: number;
    candidateK: number;
  };
  groupingMethods: GroupingMethodRecord[];
  topPureHumanCommunities: CommunityHighlight[];
  topCompactBotCommunities: CommunityHighlight[];
  takeaways: string[];
}

export interface CommunityHighlight {
  communityId: string;
  communitySize: number;
  archetype: string;
  purity: number;
  density: number;
  clusteringCoefficient: number;
  botRatio: number;
}

export interface GroupingMethodRecord {
  methodKey: string;
  methodName: string;
  communities: number;
  largestCommunity: number;
  medianCommunity: number;
  structuralEntropy: number;
  weightedModularity: number;
  weightedMeanDensity: number;
  weightedMeanClustering: number;
  weightedMeanConductance: number;
  globalPurity: number | null;
}

export interface CommunityNode {
  id: string;
  label: string;
  communitySize: number;
  density: number;
  clusteringCoefficient: number;
  purity: number;
  botRatio: number;
  averageDegree: number;
  archetype: string;
  trainCount: number;
  validCount: number;
  testCount: number;
  encodingNodeId?: string;
  encodingDepth?: number;
  x: number;
  y: number;
}

export interface CommunityEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  edgeCount: number;
  meanContentSimilarity: number | null;
  meanBehaviorSimilarity: number | null;
  meanTemporalSimilarity: number | null;
  meanNetworkSimilarity: number | null;
}

export interface CommunitySubgraphNode {
  userId: string;
  x: number;
  y: number;
}

export interface CommunitySubgraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface CommunitySubgraph {
  userIds: string[];
  nodes: CommunitySubgraphNode[];
  edges: CommunitySubgraphEdge[];
}

export interface GraphBundle {
  meta: {
    communityCount: number;
    interCommunityEdges: number;
    userCount: number;
  };
  nodes: CommunityNode[];
  edges: CommunityEdge[];
  subgraphs: Record<string, CommunitySubgraph>;
}

export interface CommunityRecord {
  communityId: string;
  communitySize: number;
  humanCount: number;
  botCount: number;
  unknownLabelCount: number;
  botRatio: number;
  purity: number;
  density: number;
  averageDegree: number;
  clusteringCoefficient: number;
  trainCount: number;
  validCount: number;
  testCount: number;
  predictedLabelByTrainMajority: string;
  labelSource: string;
  archetype: string;
  encodingNodeId?: string;
  encodingDepth?: number;
  topUserIds: string[];
}

export interface UserRecord {
  userId: string;
  username: string;
  name: string;
  split: string;
  label: string;
  communityId: string;
  communitySize: number;
  communityPurity: number;
  communityDensity: number;
  communityClustering: number;
  communityArchetype: string;
  descriptionExcerpt: string;
  tripletSummary: string;
  followersCount: number;
  followingCount: number;
  tweetsTotal: number;
  verified: number;
  canFullPipeline: number;
  canTriplet: number;
  canPostType: number;
  postTypeRatios: {
    original: number;
    retweet: number;
    commentReply: number;
    linkShare: number;
  };
}

export interface CompareSummary {
  methods: GroupingMethodRecord[];
  archetypeCounts: Record<string, number>;
  representativeCommunities: RepresentativeCommunity[];
  primaryMethodKey: string;
  purityNote: string;
}

export interface RepresentativeCommunity {
  archetype: string;
  selectionRank: number;
  communityId: string;
  communitySize: number;
  botRatio: number;
  purity: number;
  density: number;
  clusteringCoefficient: number;
  encodingDepth: number;
  predictedLabel: string;
}

export interface ErrorCase {
  userId: string;
  split: string;
  label: string;
  communityId: string;
  communitySize: number;
  baselinePredictedLabel: string;
  rerankerPredictedLabel: string;
  baselineBotScore: number;
  rerankerBotScore: number;
  scoreDelta: number;
  username: string;
  name: string;
  descriptionExcerpt: string;
  followersCount: number;
  followingCount: number;
  tweetsTotal: number;
  verified: number;
  canFullPipeline: number;
}

export interface CommunityChangeRecord {
  communityId: string;
  communitySize: number;
  changedCount: number;
  fixedCount: number;
  regressedCount: number;
  netGain: number;
  baselineErrorRate: number;
  rerankerErrorRate: number;
}

export interface ErrorsSummary {
  fixedCases: ErrorCase[];
  regressedCases: ErrorCase[];
  unchangedErrors: ErrorCase[];
  communityChanges: CommunityChangeRecord[];
}

export interface MethodSummary {
  framework: string;
  channels: {
    id: string;
    title: string;
    summary: string;
    formula: string;
  }[];
  configuration: Record<string, string | number>;
  notes: string[];
}
