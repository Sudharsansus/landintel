import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createJob, uploadFile } from '../api/client';

type FileState = { file: File; status: 'pending' | 'uploading' | 'done' | 'error'; key?: string; error?: string };

// Upload flow:
// 1. User selects/drops FMB PDFs.
// 2. Click "Upload & Process": each file is POSTed to /files/upload?job_id=<batchId>.
//    The batchId is a random hex string used only for the S3 key path — it is NOT
//    the final job ID (the server assigns that when POST /jobs is called).
// 3. Collected S3 keys are submitted to POST /jobs.
// 4. Navigate to /jobs.
//
// Note: file upload will fail if S3 is not configured (ConfigError from the API).
// This is expected behaviour — the API requires AWS credentials.

function randomHex(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export function Upload() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<FileState[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming).filter((f) => f.name.endsWith('.pdf'));
    if (arr.length === 0) return;
    setFiles((prev) => [
      ...prev,
      ...arr.map((file) => ({ file, status: 'pending' as const })),
    ]);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  }, []);

  const onInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files);
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const submit = async () => {
    if (files.length === 0) return;
    setSubmitting(true);
    setGlobalError(null);
    const batchId = randomHex();
    const updated = [...files];
    const keys: string[] = [];

    for (let i = 0; i < updated.length; i++) {
      updated[i] = { ...updated[i], status: 'uploading' };
      setFiles([...updated]);
      try {
        const result = await uploadFile(updated[i].file, batchId);
        updated[i] = { ...updated[i], status: 'done', key: result.key };
        keys.push(result.key);
      } catch (err) {
        updated[i] = { ...updated[i], status: 'error', error: String(err) };
      }
      setFiles([...updated]);
    }

    if (keys.length === 0) {
      setGlobalError('All uploads failed — check API logs and S3 configuration.');
      setSubmitting(false);
      return;
    }

    try {
      await createJob(keys);
      navigate('/jobs');
    } catch (err) {
      setGlobalError(String(err));
      setSubmitting(false);
    }
  };

  const canSubmit = files.length > 0 && !submitting && files.every((f) => f.status === 'pending');

  return (
    <div style={{ maxWidth: 640, margin: '0 auto', padding: '32px 16px' }}>
      <h2 style={{ fontSize: 18, marginBottom: 4 }}>Upload FMB Surveys</h2>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 20 }}>
        Drop one or more FMB PDF files. Files are uploaded to S3 and queued for
        automated extraction (M1 → M4). Georeferencing (M2) and village assembly (M3)
        run after extraction completes.
      </p>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        style={{
          border: `2px dashed ${dragOver ? '#1565c0' : '#ccc'}`,
          borderRadius: 4,
          padding: '40px 20px',
          textAlign: 'center',
          background: dragOver ? '#e3f2fd' : '#fafafa',
          cursor: 'pointer',
          marginBottom: 16,
          transition: 'background 0.15s, border-color 0.15s',
        }}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".pdf"
          multiple
          style={{ display: 'none' }}
          onChange={onInput}
        />
        <div style={{ fontSize: 32, marginBottom: 8 }}>📄</div>
        <div style={{ fontSize: 14, color: '#555' }}>
          Drag FMB PDFs here, or click to select
        </div>
        <div style={{ fontSize: 12, color: '#999', marginTop: 4 }}>PDF files only</div>
      </div>

      {/* Selected files */}
      {files.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          {files.map((fs, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '6px 10px',
                border: '1px solid #e0e0e0',
                borderRadius: 3,
                marginBottom: 4,
                background: fs.status === 'error' ? '#fff5f5' : '#fff',
                fontSize: 13,
              }}
            >
              <span style={{ fontFamily: 'monospace', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {fs.file.name}
              </span>
              <span style={{ marginLeft: 8, fontSize: 11, color: '#888' }}>
                {(fs.file.size / 1024).toFixed(0)} KB
              </span>
              <span style={{ marginLeft: 12, fontSize: 11, width: 70, textAlign: 'right', color:
                fs.status === 'done' ? '#1b5e20' :
                fs.status === 'error' ? '#c62828' :
                fs.status === 'uploading' ? '#e65100' : '#888'
              }}>
                {fs.status === 'error' ? 'failed' : fs.status}
              </span>
              {fs.status === 'pending' && (
                <button
                  onClick={(e) => { e.stopPropagation(); removeFile(idx); }}
                  style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: '#999', fontSize: 16 }}
                >
                  ×
                </button>
              )}
            </div>
          ))}
          {files.some((f) => f.status === 'error') && (
            <div style={{ fontSize: 12, color: '#c62828', marginTop: 4 }}>
              {files.filter((f) => f.status === 'error').map((f, i) => (
                <div key={i}>{f.file.name}: {f.error}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {globalError && (
        <div style={{ padding: '8px 12px', background: '#ffe0e0', border: '1px solid #ef9a9a', borderRadius: 3, fontSize: 13, marginBottom: 12, color: '#7f0000' }}>
          {globalError}
        </div>
      )}

      <button
        onClick={submit}
        disabled={!canSubmit}
        style={{
          padding: '10px 24px',
          background: canSubmit ? '#1565c0' : '#bdbdbd',
          color: '#fff',
          border: 'none',
          borderRadius: 3,
          cursor: canSubmit ? 'pointer' : 'not-allowed',
          fontSize: 14,
          fontWeight: 600,
        }}
      >
        {submitting ? 'Uploading…' : `Upload & Process (${files.length} file${files.length !== 1 ? 's' : ''})`}
      </button>
    </div>
  );
}
