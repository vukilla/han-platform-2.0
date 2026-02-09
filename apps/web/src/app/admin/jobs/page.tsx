"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { JobFailureOut, listFailedJobs } from "@/lib/api";

export default function AdminJobsPage() {
  const [jobs, setJobs] = useState<JobFailureOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listFailedJobs()
      .then(setJobs)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <p className="section-eyebrow">Admin</p>
          <h1 className="text-3xl font-semibold text-black">Failed jobs</h1>
        </div>
        <Badge label="Ops" tone="amber" />
      </header>

      <Card>
        {error && <p className="text-sm text-red-600">Error: {error}</p>}
        {!error && jobs.length === 0 && (
          <p className="text-sm text-black/60">No failed or retrying jobs found.</p>
        )}
        {jobs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="border-b border-black/10 text-left text-black/60">
                <tr>
                  <th className="py-3 pr-4">Type</th>
                  <th className="py-3 pr-4">Job</th>
                  <th className="py-3 pr-4">Status</th>
                  <th className="py-3 pr-4">Error</th>
                  <th className="py-3 pr-4">Logs</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id} className="border-b border-black/5">
                    <td className="py-3 pr-4 uppercase text-black/70">{job.job_type}</td>
                    <td className="py-3 pr-4 font-mono text-xs">{job.id}</td>
                    <td className="py-3 pr-4">
                      <Badge label={job.status} tone={job.status === "FAILED" ? "rose" : "amber"} />
                    </td>
                    <td className="py-3 pr-4 text-black/70">{job.error ?? "—"}</td>
                    <td className="py-3 pr-4">
                      {job.logs_uri ? (
                        <a className="text-black underline" href={job.logs_uri} target="_blank" rel="noreferrer">
                          Logs
                        </a>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
