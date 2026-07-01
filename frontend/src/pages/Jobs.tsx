import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listJobs, type Job } from '../api/client';
import { JobCard } from '../components/JobCard';

const ACTIVE_STATUSES = new Set(['queued', 'running']);
const POLL_INTERVAL_MS = 5000;

export function Jobs() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetch = useCallback(async () => {
    try {
      const { items } = await listJobs();
      setJobs(items);
      setError(null);
      // Stop polling when no active jobs remain.
      const hasActive = items.some((j) => ACTIVE_STATUSES.has(j.status));
      if (!hasActive && pollRef.current !== null) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
    pollRef.current = setInterval(fetch, POLL_INTERVAL_MS);
    return () => {
      if (pollRef.current !== null) clearInterval(pollRef.current);
    };
  }, [fetch]);

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '32px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ fontSize: 18 }}>Conversion Jobs</h2>
        <button
          onClick={() => navigate('/upload')}
          style={{ padding: '8px 16px', background: '#1565c0', color: '#fff', border: 'none', borderRadius: 3, cursor: 'pointer', fontSize: 13 }}
        >
          + New job
        </button>
      </div>

      {loading && <p style={{ color: '#888', fontSize: 13 }}>Loading…</p>}
      {error && (
        <div style={{ padding: '8px 12px', background: '#ffe0e0', border: '1px solid #ef9a9a', borderRadius: 3, fontSize: 13, color: '#7f0000', marginBottom: 12 }}>
          {error}
        </div>
      )}

      {!loading && !error && jobs.length === 0 && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#9e9e9e' }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>📋</div>
          <p style={{ fontSize: 14 }}>No jobs yet. Upload FMB PDFs to start.</p>
        </div>
      )}

      {jobs.map((job) => (
        <JobCard key={job.id} job={job} onClick={() => navigate(`/jobs/${job.id}`)} />
      ))}
    </div>
  );
}
