import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import { DefinitionList, EmptyState, ErrorBanner, LoadingState, PageHeader, Panel, StatusBadge, formatDate } from "../components/ui";

export function ProjectWorkspacePage() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const fileInput = useRef<HTMLInputElement>(null);
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const paper = useQuery({ queryKey: ["paper", projectId], queryFn: () => api.paper(projectId) });
  const submission = useQuery({ queryKey: ["submission", projectId], queryFn: () => api.submission(projectId) });
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["project", projectId] }),
      queryClient.invalidateQueries({ queryKey: ["paper", projectId] }),
      queryClient.invalidateQueries({ queryKey: ["submission", projectId] }),
    ]);
  };
  const run = useMutation({ mutationFn: (action: () => Promise<unknown>) => action(), onSuccess: refresh });
  const upload = useMutation({ mutationFn: (file: File) => api.uploadProjectFile(projectId, file), onSuccess: refresh });

  if (project.isLoading) return <LoadingState label="Loading project state…" />;
  if (project.error || !project.data) return <ErrorBanner error={project.error ?? new Error("Project not found")} />;
  const state = project.data;

  return (
    <>
      <PageHeader
        title={state.title}
        description={`Project ${state.project_id} · ProblemState v${state.version}`}
        actions={<Link className="button" to="/projects">Back to projects</Link>}
      />
      {run.error ? <div className="mb-5"><ErrorBanner error={run.error} /></div> : null}
      {upload.error ? <div className="mb-5"><ErrorBanner error={upload.error} /></div> : null}
      <div className="grid gap-5 xl:grid-cols-[1fr_320px]">
        <div className="space-y-5">
          <Panel title="Problem">
            <p className="whitespace-pre-wrap text-sm leading-6">{state.raw_problem}</p>
          </Panel>
          <Panel title="Attachments" description="Original files remain backend-managed and immutable.">
            <input
              ref={fileInput}
              className="input w-full"
              type="file"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) upload.mutate(file);
              }}
            />
            {state.registered_files?.length ? (
              <ul className="mt-4 space-y-2 text-sm">
                {state.registered_files.map((file) => <li key={file.file_id} className="rounded-md bg-slate-50 p-3">{file.original_name} · {file.media_type}</li>)}
              </ul>
            ) : <div className="mt-4"><EmptyState>No files registered.</EmptyState></div>}
          </Panel>
          <Panel title="Backend workflows" description="Each action calls one existing backend workflow endpoint; the browser never sequences Agents.">
            <div className="flex flex-wrap gap-2">
              <WorkflowButton label="Run reasoning" pending={run.isPending} action={() => api.runReasoning(projectId)} run={run.mutate} />
              <WorkflowButton label="Analyze data" pending={run.isPending} action={() => api.analyzeData(projectId)} run={run.mutate} />
              <WorkflowButton label="Build & solve" pending={run.isPending} action={() => api.runMathematical(projectId)} run={run.mutate} />
              <WorkflowButton label="Verify" pending={run.isPending} action={() => api.runVerification(projectId)} run={run.mutate} />
              <WorkflowButton label="Build paper" pending={run.isPending} action={() => api.runPaper(projectId)} run={run.mutate} />
            </div>
          </Panel>
        </div>
        <div className="space-y-5">
          <Panel title="Current state">
            <DefinitionList items={[
              ["Stage", <StatusBadge value={state.current_stage} />],
              ["Status", <StatusBadge value={state.status} />],
              ["Updated", formatDate(state.updated_at)],
              ["Data stage", <StatusBadge value={state.data_stage} />],
              ["Selected model", state.selected_model?.name ?? "Not selected"],
              ["Verified result", state.verified_result_id ?? "None"],
            ]} />
          </Panel>
          <Panel title="Paper & submission">
            <DefinitionList items={[
              ["Paper", paper.data ? `v${paper.data.version}` : "Not available"],
              ["Paper status", paper.data?.status ? <StatusBadge value={paper.data.status} /> : "Not exposed"],
              ["Submission", submission.data ? <StatusBadge value={submission.data.status} /> : "Not available"],
              ["Final check", "Requires exact backend profile and paper version"],
            ]} />
          </Panel>
        </div>
      </div>
    </>
  );
}

function WorkflowButton({ label, pending, action, run }: { label: string; pending: boolean; action: () => Promise<unknown>; run: (action: () => Promise<unknown>) => void }) {
  return <button className="button" disabled={pending} onClick={() => run(action)}>{pending ? "Running…" : label}</button>;
}
