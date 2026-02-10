import { getToken } from "./auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }

  return response.json() as Promise<T>;
}

export type LoginResponse = {
  user: { id: string; email: string; name?: string | null };
  token: string;
};

export async function login(email: string, name?: string) {
  return apiFetch<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, name }),
  });
}

export async function createProject(name: string, description?: string) {
  return apiFetch<{ id: string; name: string }>("/projects", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export async function listProjects() {
  return apiFetch<Array<{ id: string; name: string }>>("/projects");
}

export async function createDemo(projectId: string, robotModel?: string, objectId?: string) {
  return apiFetch<{ id: string }>("/demos", {
    method: "POST",
    body: JSON.stringify({ project_id: projectId, robot_model: robotModel, object_id: objectId }),
  });
}

export type DemoOut = {
  id: string;
  project_id: string;
  uploader_id?: string | null;
  video_uri?: string | null;
  fps?: number | null;
  duration?: number | null;
  robot_model?: string | null;
  object_id?: string | null;
  status: string;
};

export async function getDemo(demoId: string) {
  return apiFetch<DemoOut>(`/demos/${demoId}`);
}

export async function getDemoUploadUrl(demoId: string) {
  return apiFetch<{ upload_url: string; video_uri: string }>(`/demos/${demoId}/upload-url`, {
    method: "POST",
  });
}

export async function uploadDemoVideo(demoId: string, file: File) {
  const token = getToken();
  const headers = new Headers();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_URL}/demos/${demoId}/upload`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<DemoOut>;
}

export async function annotateDemo(
  demoId: string,
  payload: {
    ts_contact_start: number;
    ts_contact_end: number;
    anchor_type: string;
    key_bodies?: string[];
    notes?: string;
  },
) {
  return apiFetch(`/demos/${demoId}/annotations`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runXgen(demoId: string, params_json?: Record<string, unknown>) {
  return apiFetch<XGenJobOut>(`/demos/${demoId}/xgen/run`, {
    method: "POST",
    body: JSON.stringify({ params_json }),
  });
}

export type XGenJobOut = {
  id: string;
  demo_id: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  params_json?: Record<string, unknown> | null;
  logs_uri?: string | null;
  error?: string | null;
  idempotency_key?: string | null;
};

export async function getXgenJob(jobId: string) {
  return apiFetch<XGenJobOut>(`/xgen/jobs/${jobId}`);
}

export type DatasetOut = {
  id: string;
  project_id: string;
  source_demo_id?: string | null;
  version: number;
  status: string;
  summary_json?: Record<string, unknown> | null;
};

export type DatasetClipOut = {
  clip_id: string;
  dataset_id: string;
  uri_npz: string;
  uri_preview_mp4?: string | null;
  augmentation_tags?: string[] | null;
  stats_json?: Record<string, unknown> | null;
};

export async function getDataset(datasetId: string) {
  return apiFetch<DatasetOut>(`/datasets/${datasetId}`);
}

export async function listDatasetClips(datasetId: string) {
  return apiFetch<DatasetClipOut[]>(`/datasets/${datasetId}/clips`);
}

export async function getDatasetDownloadUrl(datasetId: string) {
  return apiFetch<{ download_url: string }>(`/datasets/${datasetId}/download-url`);
}

export async function listDatasets(projectId?: string) {
  const query = projectId ? `?project_id=${projectId}` : "";
  return apiFetch<DatasetOut[]>(`/datasets${query}`);
}

export async function runXmimic(datasetId: string, mode: "nep" | "mocap", params_json?: Record<string, unknown>) {
  return apiFetch<XMimicJobOut>(`/datasets/${datasetId}/xmimic/run`, {
    method: "POST",
    body: JSON.stringify({ mode, params_json }),
  });
}

export type XMimicJobOut = {
  id: string;
  dataset_id: string;
  mode: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  params_json?: Record<string, unknown> | null;
  logs_uri?: string | null;
  error?: string | null;
  idempotency_key?: string | null;
};

export async function getXmimicJob(jobId: string) {
  return apiFetch<XMimicJobOut>(`/xmimic/jobs/${jobId}`);
}

export async function listXmimicJobs(datasetId?: string) {
  const query = datasetId ? `?dataset_id=${datasetId}` : "";
  return apiFetch<XMimicJobOut[]>(`/xmimic/jobs${query}`);
}

export type PolicyOut = {
  id: string;
  xmimic_job_id: string;
  checkpoint_uri: string;
  exported_at?: string | null;
  metadata_json?: Record<string, unknown> | null;
};

export async function listPolicies(xmimicJobId?: string) {
  const query = xmimicJobId ? `?xmimic_job_id=${xmimicJobId}` : "";
  return apiFetch<PolicyOut[]>(`/policies${query}`);
}

export async function getPolicy(policyId: string) {
  return apiFetch<PolicyOut>(`/policies/${policyId}`);
}

export type EvalRunOut = {
  id: string;
  policy_id: string;
  env_task: string;
  sr?: number | null;
  gsr?: number | null;
  eo?: number | null;
  eh?: number | null;
  report_uri?: string | null;
  videos_uri?: string | null;
};

export async function getEval(evalId: string) {
  return apiFetch<EvalRunOut>(`/eval/${evalId}`);
}

export async function listEvalRuns(policyId?: string) {
  const query = policyId ? `?policy_id=${policyId}` : "";
  return apiFetch<EvalRunOut[]>(`/eval${query}`);
}

export type RewardEventOut = {
  id: string;
  user_id: string;
  entity_type: string;
  entity_id: string;
  points: number;
  reason: string;
  created_at: string;
};

export async function getRewardsMe() {
  return apiFetch<RewardEventOut[]>(`/rewards/me`);
}

export type QualityScoreOut = {
  id: string;
  entity_type: string;
  entity_id: string;
  score?: number | null;
  breakdown_json?: Record<string, unknown> | null;
  validator_status?: string | null;
  validator_notes?: string | null;
  validator_id?: string | null;
  validated_at?: string | null;
};

export async function getQualityScore(entityType: string, entityId: string) {
  return apiFetch<QualityScoreOut>(`/quality/${entityType}/${entityId}`);
}

export async function reviewQualityScore(
  entityType: string,
  entityId: string,
  payload: { status: string; notes?: string },
) {
  return apiFetch<QualityScoreOut>(`/quality/${entityType}/${entityId}/review`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export type JobFailureOut = {
  job_type: "xgen" | "xmimic";
  id: string;
  status: string;
  error?: string | null;
  logs_uri?: string | null;
  demo_id?: string | null;
  dataset_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
};

export async function listFailedJobs() {
  return apiFetch<JobFailureOut[]>("/ops/jobs/failed");
}
