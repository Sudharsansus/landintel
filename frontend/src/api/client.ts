// All types mirror the FastAPI schemas exactly.
// If the API contract changes, update the corresponding Python schema first.

export type JobStatus =
  | 'queued'
  | 'running'
  | 'needs_review'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type Stage =
  | 'intake'
  | 'extract'
  | 'georef'
  | 'assemble'
  | 'report'
  | 'delivered';

export type PlotStatus =
  | 'extracted'
  | 'validated'
  | 'flagged'
  | 'corrected'
  | 'georeferenced'
  | 'assembled'
  | 'failed';

export interface Job {
  id: string;
  client_id: string;
  status: JobStatus;
  stage: Stage;
  input_files: string[];
  output_files: string[];
  audit: string[];
  created_at: string;
}

export interface JobList {
  items: Job[];
  total: number;
}

export interface PlotSummary {
  survey_no: string;
  status: PlotStatus;
  stated_area: number | null;
  flags: string[];
}

export interface PlotDetail extends PlotSummary {
  district: string;
  taluk: string;
  village: string;
  scale: number | null;
  measurement_count: number;
  corner_count: number;
  boundary_closed: boolean | null;
}

export interface UploadResult {
  key: string;
  filename: string;
  size: number;
}

export interface CorrectionPayload {
  field: string;
  old: string;
  new: string;
  measurement_ref?: string;
}

export interface JobArtifact {
  stage: string;
  filename: string;
  url: string;
}

export interface CorrectionResult {
  id: string;
  plot_id: string;
  field: string;
  old: string;
  new: string;
}

// In dev (no env var): falls back to '/api', which Vite proxies to localhost:8000.
// In prod build: set VITE_API_BASE_URL=https://landintel-api.onrender.com
const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${init?.method ?? 'GET'} ${path}: ${res.status} ${text}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Jobs ────────────────────────────────────────────────────────────────────

export function listJobs(limit = 20, skip = 0): Promise<JobList> {
  return apiFetch<JobList>(`/jobs?limit=${limit}&skip=${skip}`);
}

export function getJob(id: string): Promise<Job> {
  return apiFetch<Job>(`/jobs/${id}`);
}

export function createJob(inputFiles: string[]): Promise<Job> {
  return apiFetch<Job>('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input_files: inputFiles }),
  });
}

export function cancelJob(id: string): Promise<void> {
  return apiFetch<void>(`/jobs/${id}`, { method: 'DELETE' });
}

// ── File upload ─────────────────────────────────────────────────────────────

export async function uploadFile(file: File, jobId: string): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  // No Content-Type header: browser sets multipart boundary automatically.
  const res = await fetch(
    `${BASE}/files/upload?job_id=${encodeURIComponent(jobId)}`,
    { method: 'POST', body: form },
  );
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Upload ${file.name}: ${res.status} ${text}`);
  }
  return res.json() as Promise<UploadResult>;
}

export function getJobFiles(jobId: string): Promise<JobArtifact[]> {
  return apiFetch<JobArtifact[]>(`/jobs/${jobId}/files`);
}

// ── Review ──────────────────────────────────────────────────────────────────

export function listFlaggedPlots(): Promise<PlotSummary[]> {
  return apiFetch<PlotSummary[]>('/review/flagged');
}

export function getPlotDetail(surveyNo: string): Promise<PlotDetail> {
  return apiFetch<PlotDetail>(`/review/${encodeURIComponent(surveyNo)}`);
}

export function submitCorrection(
  surveyNo: string,
  jobId: string,
  payload: CorrectionPayload,
): Promise<CorrectionResult> {
  return apiFetch<CorrectionResult>(
    `/review/${encodeURIComponent(surveyNo)}/corrections?job_id=${encodeURIComponent(jobId)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  );
}
