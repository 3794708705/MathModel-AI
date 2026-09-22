import { createBrowserRouter } from "react-router-dom";

import { AppShell } from "./AppShell";
import { BenchmarkPage } from "../pages/BenchmarkPage";
import { DashboardPage } from "../pages/DashboardPage";
import { ModelsApiPage } from "../pages/ModelsApiPage";
import { ProjectsPage } from "../pages/ProjectsPage";
import { ProjectWorkspacePage } from "../pages/ProjectWorkspacePage";
import { RoutingPage } from "../pages/RoutingPage";
import { SystemPage } from "../pages/SystemPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "projects", element: <ProjectsPage /> },
      { path: "projects/:projectId", element: <ProjectWorkspacePage /> },
      { path: "models", element: <ModelsApiPage /> },
      { path: "routing", element: <RoutingPage /> },
      { path: "benchmark", element: <BenchmarkPage /> },
      { path: "system", element: <SystemPage /> },
      { path: "*", element: <p className="rounded-md border bg-white p-6">Page not found.</p> },
    ],
  },
]);
