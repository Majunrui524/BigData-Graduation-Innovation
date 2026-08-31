import { Route, Routes } from "react-router-dom";

import { AppShell } from "./AppShell";
import { CommunitiesPage } from "../pages/CommunitiesPage";
import { ComparePage } from "../pages/ComparePage";
import { ErrorsPage } from "../pages/ErrorsPage";
import { GraphPage } from "../pages/GraphPage";
import { OverviewPage } from "../pages/OverviewPage";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/communities" element={<CommunitiesPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/errors" element={<ErrorsPage />} />
      </Routes>
    </AppShell>
  );
}
