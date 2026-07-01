import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { cancelJob, getJob, getJobFiles, type Job, type JobArtifact } from '../api/client';
import { StageProgress } from '../components/StageProgress';

// JobDetail polls while the job is active and stops when it reaches a terminal state.
// It does NOT show per-plot status — JobResponse does not include plots.
// Per-plot review is on the Review page (/review).

const ACTIVE_STATUSES = new Set(['queued', 'running']);
const POLL_MS = 4000;

export function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [files, setFiles] = useState<JobArtifact[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (!id) return;
    let active = true;

    async function load() {
      try {
        const j = await getJob(id!);
        if (active) setJob(j);
        // Fetch presigned download URLs whenever the job is no longer active.
        if (active && !ACTIVE_STATUSES.has(j.status)) {
          getJobFiles(id!).then((f) => { if (active) setFiles(f); }).catch(() => {});
        }
        if (active && ACTIVE_STATUSES.has(j.status)) {
          setTimeout(load, POLL_MS);
        }
      } catch (err) {
        if (active) setError(String(err));
      }
    }

    load();
    return () => { active = false; };
  }, [id]);

  const handleCancel = async () => {
    if (!job) return;
    setCancelling(true);
    try {
      await cancelJob(job.id);
      setJob((j) => j ? { ...j, status: 'cancelled' } : j);
    } catch (err) {
      setError(String(err));
    } finally {
      setCancelling(false);
    }
  };

  if (error) return (
    <div style={{ padding: 32 }}>
      <button onClick={() => navigate('/jobs')} style={backStyle}>← Back</button>
      <div style={{ color: '#c62828', marginTop: 16 }}>{error}</div>
    </div>
  );

  if (!job) return <div style={{ padding: 32, color: '#888' }}>Loading…</div>;

  const isActive = ACTIVE_STATUSES.has(job.status);
  const needsReview = job.status === 'needs_review';

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '32px 16px' }}>
      <button onClick={() => navigate('/jobs')} style={backStyle}>← All jobs</button>

      <div style={{ marginTop: 20, marginBottom: 4, display: 'flex', alignItems: 'baseline', gap: 12 }}>
        <h2 style={{ fontSize: 18 }}>Job</h2>
        <span style={{ fontFamily: 'monospace', fontSize: 14, color: '#666' }}>{job.id}</span>
      </div>

      <div style={{ fontSize: 12, color: '#888', marginBottom: 20 }}>
        Created {new Date(job.created_at).toLocaleString()}
        {' · '}{job.input_files.length} input file{job.input_files.length !== 1 ? 's' : ''}
      </div>

      <div style={{ marginBottom: 20 }}>
        <StageProgress current={job.stage} />
      </div>

      {/* Status + actions */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 20, flexWrap: 'wrap' }}>
        <StatusBadge status={job.status} />
        {isActive && (
          <button onClick={handleCancel} disabled={cancelling} style={dangerStyle}>
            {cancelling ? 'Cancelling…' : 'Cancel job'}
          </button>
        )}
        {needsReview && (
          <button onClick={() => navigate('/review')} style={reviewStyle}>
            Review flagged plots →
          </button>
        )}
      </div>

      {/* Output files — presigned S3 download links, available once job finishes */}
      {files.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Downloads</h3>
          {files.map((f, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <span style={{
                fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                background: '#e3f2fd', color: '#1565c0', padding: '2px 7px', borderRadius: 3,
                minWidth: 60, textAlign: 'center',
              }}>
                {f.stage}
              </span>
              <a
                href={f.url}
                download={f.filename}
                style={{ fontSize: 13, color: '#1565c0', fontFamily: 'monospace' }}
              >
                {f.filename}
              </a>
            </div>
          ))}
        </div>
      )}

      {/* Audit trail */}
      {job.audit.length > 0 && (
        <div>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Audit trail</h3>
          <div style={{ border: '1px solid #e0e0e0', borderRadius: 3, background: '#fafafa', padding: '8px 12px' }}>
            {job.audit.map((entry, i) => (
              <div key={i} style={{ fontSize: 12, fontFamily: 'monospace', padding: '3px 0', borderBottom: i < job.audit.length - 1 ? '1px solid #eee' : 'none', color: '#444' }}>
                {entry}
              </div>
            ))}
          </div>
        </div>
      )}

      {isActive && (
        <div style={{ marginTop: 16, fontSize: 12, color: '#888' }}>
          Polling every {POLL_MS / 1000}s…
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, React.CSSProperties> = {
    queued:       { background: '#e3f2fd', color: '#1565c0', border: '1px solid #90caf9' },
    running:      { background: '#fff8e1', color: '#e65100', border: '1px solid #ffcc02' },
    needs_review: { background: '#fff2cc', color: '#7a5900', border: '1px solid #e6c62b' },
    completed:    { background: '#e8f5e9', color: '#1b5e20', border: '1px solid #81c784' },
    failed:       { background: '#ffe0e0', color: '#7f0000', border: '1px solid #ef9a9a' },
    cancelled:    { background: '#f5f5f5', color: '#616161', border: '1px solid #bdbdbd' },
  };
  return (
    <span style={{ ...(styles[status] ?? {}), padding: '4px 12px', borderRadius: 3, fontSize: 13, fontWeight: 600, textTransform: 'capitalize' }}>
      {status.replace('_', ' ')}
    </span>
  );
}

const backStyle: React.CSSProperties = { background: 'none', border: 'none', cursor: 'pointer', color: '#1565c0', fontSize: 13, padding: 0 };
const dangerStyle: React.CSSProperties = { padding: '5px 12px', background: 'none', border: '1px solid #ef9a9a', color: '#c62828', borderRadius: 3, cursor: 'pointer', fontSize: 13 };
const reviewStyle: React.CSSProperties = { padding: '5px 12px', background: '#fff2cc', border: '1px solid #e6c62b', color: '#7a5900', borderRadius: 3, cursor: 'pointer', fontSize: 13, fontWeight: 600 };
