import type { Job, JobStatus } from '../api/client';

// Status badge colours match the PDF area statement palette (area_statement.py).
const STATUS_STYLE: Record<JobStatus, React.CSSProperties> = {
  queued:       { background: '#e3f2fd', color: '#1565c0', border: '1px solid #90caf9' },
  running:      { background: '#fff8e1', color: '#e65100', border: '1px solid #ffcc02' },
  needs_review: { background: '#fff2cc', color: '#7a5900', border: '1px solid #e6c62b' },
  completed:    { background: '#e8f5e9', color: '#1b5e20', border: '1px solid #81c784' },
  failed:       { background: '#ffe0e0', color: '#7f0000', border: '1px solid #ef9a9a' },
  cancelled:    { background: '#f5f5f5', color: '#616161', border: '1px solid #bdbdbd' },
};

interface Props {
  job: Job;
  onClick: () => void;
}

export function JobCard({ job, onClick }: Props) {
  const badge = STATUS_STYLE[job.status];
  const date = new Date(job.created_at).toLocaleString();
  const shortId = job.id.slice(0, 12);

  return (
    <div
      onClick={onClick}
      style={{
        border: '1px solid #d0d0d0',
        borderRadius: 3,
        padding: '12px 16px',
        marginBottom: 8,
        cursor: 'pointer',
        background: '#fff',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 12,
      }}
    >
      <div>
        <span style={{ fontFamily: 'monospace', fontSize: 13, color: '#444' }}>{shortId}…</span>
        <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>{date}</div>
        <div style={{ fontSize: 12, color: '#888' }}>
          {job.input_files.length} file{job.input_files.length !== 1 ? 's' : ''} — stage: {job.stage}
        </div>
      </div>
      <span
        style={{
          ...badge,
          padding: '3px 10px',
          borderRadius: 3,
          fontSize: 12,
          fontWeight: 600,
          whiteSpace: 'nowrap',
          textTransform: 'capitalize',
        }}
      >
        {job.status.replace('_', ' ')}
      </span>
    </div>
  );
}
