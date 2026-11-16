import { Route, Routes } from "react-router-dom";
import { ActiveReconstructionProvider } from "./hooks/useActiveReconstruction";
import { AppLayout } from "./layouts/AppLayout";
import { AnalysisPage } from "./pages/AnalysisPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { FindingsPage } from "./pages/FindingsPage";
import { GenerationPage } from "./pages/GenerationPage";
import { MethodologyPage } from "./pages/MethodologyPage";
import { NewReconstructionPage } from "./pages/NewReconstructionPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PresentationPage } from "./pages/PresentationPage";
import { PromptStudioPage } from "./pages/PromptStudioPage";
import { RefinementPage } from "./pages/RefinementPage";

export default function App() {
  return (
    <ActiveReconstructionProvider>
      <Routes>
        <Route path="/present" element={<PresentationPage />} />
        <Route element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="/new" element={<NewReconstructionPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/prompts" element={<PromptStudioPage />} />
          <Route path="/generation" element={<GenerationPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/refinement" element={<RefinementPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/findings" element={<FindingsPage />} />
          <Route path="/methodology" element={<MethodologyPage />} />
          <Route path="*" element={<OverviewPage />} />
        </Route>
      </Routes>
    </ActiveReconstructionProvider>
  );
}
