import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { EmptyState, ErrorBanner, LoadingState, PageHeader, Panel, StatusBadge, formatDate } from "../components/ui";

export function ProjectsPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ name: "", title: "", raw_problem: "", competition: "" });
  const create = useMutation({
    mutationFn: api.createProject,
    onSuccess: async (project) => {
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
      setForm({ name: "", title: "", raw_problem: "", competition: "" });
      setShowCreate(false);
      void navigate(`/projects/${project.project_id}`);
    },
  });

  return (
    <>
      <PageHeader
        title="Projects"
        description="Create a competition problem and open its backend-owned ProblemState."
        actions={<button className="button button-primary" onClick={() => setShowCreate((value) => !value)}>Create project</button>}
      />
      {create.error ? <div className="mb-5"><ErrorBanner error={create.error} /></div> : null}
      {showCreate ? (
        <Panel title="New project" className="mb-5">
          <form
            className="grid gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              create.mutate({ ...form, competition: form.competition || null });
            }}
          >
            <div className="grid gap-4 md:grid-cols-2">
              <label className="field">Name<input className="input" required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>
              <label className="field">Problem title<input className="input" required value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} /></label>
            </div>
            <label className="field">Competition<input className="input" value={form.competition} onChange={(event) => setForm({ ...form, competition: event.target.value })} /></label>
            <label className="field">Problem statement<textarea className="input min-h-36" required minLength={20} value={form.raw_problem} onChange={(event) => setForm({ ...form, raw_problem: event.target.value })} /></label>
            <div><button className="button button-primary" disabled={create.isPending}>{create.isPending ? "Creating…" : "Create and open"}</button></div>
          </form>
        </Panel>
      ) : null}
      {projects.isLoading ? <LoadingState /> : projects.error ? <ErrorBanner error={projects.error} /> : projects.data?.length ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>Name</th><th>Status</th><th>Current stage</th><th>Updated</th><th /></tr></thead>
            <tbody>
              {projects.data.map((project) => (
                <tr key={project.project_id}>
                  <td><p className="font-semibold">{project.name}</p><p className="mt-1 text-xs text-muted">{project.title}</p></td>
                  <td><StatusBadge value={project.status} /></td>
                  <td><StatusBadge value={project.current_stage} /></td>
                  <td>{formatDate(project.updated_at)}</td>
                  <td className="text-right"><Link className="button" to={`/projects/${project.project_id}`}>Open</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <EmptyState>No projects are registered.</EmptyState>}
    </>
  );
}
