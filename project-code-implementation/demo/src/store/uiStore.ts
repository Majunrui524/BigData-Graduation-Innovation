import { create } from "zustand";

import type { ScoreMode } from "../lib/types";

interface UiState {
  scoreMode: ScoreMode;
  selectedCommunityId: string | null;
  selectedUserId: string | null;
  splitFilter: "all" | "train" | "valid" | "test";
  setScoreMode: (mode: ScoreMode) => void;
  setSelectedCommunityId: (id: string | null) => void;
  setSelectedUserId: (id: string | null) => void;
  setSplitFilter: (value: "all" | "train" | "valid" | "test") => void;
}

export const useUiStore = create<UiState>((set) => ({
  scoreMode: "density",
  selectedCommunityId: null,
  selectedUserId: null,
  splitFilter: "all",
  setScoreMode: (scoreMode) => set({ scoreMode }),
  setSelectedCommunityId: (selectedCommunityId) => set({ selectedCommunityId }),
  setSelectedUserId: (selectedUserId) => set({ selectedUserId }),
  setSplitFilter: (splitFilter) => set({ splitFilter }),
}));
