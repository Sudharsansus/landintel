import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  getPlotDetail,
  listFlaggedPlots,
  submitCorrection,
  type CorrectionPayload,
  type PlotDetail,
  type PlotSummary,
} from '../api/client';
import { PlotViewer } from '../components/PlotViewer';

// Review page: shows the human-review queue from anomaly.py.
// Corrections are submitted to POST /review/{survey_no}/corrections?job_id=.
// job_id is read from the URL query string (?job_id=...) — pass it when
// navigating from a specific job's detail page. Without it, correction
// submission is disabled and a warning is shown.

export function Review() {
  const [params] = useSearchParams();
  const jobId = params.get('job_id') ?? '';

  const [plots, setPlots] = useState<PlotSummary[]>([]);
  const [selected, setSelected] = useState<PlotDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [correction, setCorrection] = useState<CorrectionPayload>({ field: '', old: '', new: '' });
  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    listFlaggedPlots()
      .then(setPlots)
      .catch((e) => setListError(String(e)))
      .finally(() => setLoadingList(false));
  }, []);

  async function selectPlot(summary: PlotSummary) {
    setSelected(null);
    setSubmitResult(null);
    setSubmitError(null);
    setCorrection({ field: '', old: '', new: '' });
    setLoadingDetail(true);
    try {
      const detail = await getPlotDetail(summary.survey_no);
      setSelected(detail);
    } catch (e) {
      setListError(String(e));
    } finally {
      setLoadingDetail(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selected || !jobId) return;
    setSubmitting(true);
    setSubmitError(null);
    setSubmitResult(null);
    try {
      const result = await submitCorrection(selected.survey_no, jobId, correction);
      setSubmitResult(`Correction ${result.id.slice(0, 8)} logged — ${result.field}: "${result.old}" → "${result.new}"`);
      setCorrection({ field: '', old: '', new: '' });
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 16px', display: 'flex', gap: 24 }}>
      {/* Left: flagged plot list */}
      <div style={{ width: 260, flexShrink: 0 }}>
        <h2 style={{ fontSize: 16, marginBottom: 12 }}>Flagged plots</h2>
        {loadingList && <p style={{ fontSize: 13, color: '#888' }}>Loading…</p>}
        {listError && <p style={{ fontSize: 13, color: '#c62828' }}>{listError}</p>}
        {!loadingList && plots.length === 0 && (
          <p style={{ fontSize: 13, color: '#888' }}>No plots awaiting review.</p>
        )}
        {plots.map((p) => (
          <div
            key={p.survey_no}
            onClick={() => selectPlot(p)}
            style={{
              padding: '8px 10px',
              border: '1px solid #e0e0e0',
              borderRadius: 3,
              marginBottom: 6,
              cursor: 'pointer',
              background: selected?.survey_no === p.survey_no ? '#e3f2fd' : '#fff',
              fontSize: 13,
            }}
          >
            <div style={{ fontWeight: 700 }}>Survey {p.survey_no}</div>
            <div style={{ fontSize: 11, color: '#888' }}>
              {p.flags.length} flag{p.flags.length !== 1 ? 's' : ''}
              {p.stated_area != null ? ` · ${p.stated_area} ha` : ''}
            </div>
          </div>
        ))}
      </div>

      {/* Right: plot detail + correction form */}
      <div style={{ flex: 1 }}>
        {loadingDetail && <p style={{ color: '#888', fontSize: 13 }}>Loading plot…</p>}

        {!loadingDetail && selected && (
          <>
            <PlotViewer plot={selected} />

            <div style={{ marginTop: 20 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 10 }}>Submit correction</h3>

              {!jobId && (
                <div style={{ padding: '8px 12px', background: '#fff8e1', border: '1px solid #ffcc02', borderRadius: 3, fontSize: 12, color: '#e65100', marginBottom: 12 }}>
                  No job_id in URL — correction submission disabled. Navigate here from a
                  specific job (/jobs/:id → "Review flagged plots") to enable submissions.
                </div>
              )}

              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div>
                  <label style={labelStyle}>Field being corrected</label>
                  <input
                    style={inputStyle}
                    placeholder="e.g. measurement, stated_area"
                    value={correction.field}
                    onChange={(e) => setCorrection({ ...correction, field: e.target.value })}
                    required
                  />
                </div>
                <div>
                  <label style={labelStyle}>Old value (as OCR read it)</label>
                  <input
                    style={inputStyle}
                    placeholder="e.g. 44,2"
                    value={correction.old}
                    onChange={(e) => setCorrection({ ...correction, old: e.target.value })}
                    required
                  />
                </div>
                <div>
                  <label style={labelStyle}>Correct value</label>
                  <input
                    style={inputStyle}
                    placeholder="e.g. 44.2"
                    value={correction.new}
                    onChange={(e) => setCorrection({ ...correction, new: e.target.value })}
                    required
                  />
                </div>
                <div>
                  <label style={labelStyle}>Measurement ref (optional)</label>
                  <input
                    style={inputStyle}
                    placeholder="Measurement.line_ref if correcting a specific measurement"
                    value={correction.measurement_ref ?? ''}
                    onChange={(e) => setCorrection({ ...correction, measurement_ref: e.target.value || undefined })}
                  />
                </div>
                <button
                  type="submit"
                  disabled={submitting || !jobId}
                  style={{
                    padding: '8px 20px',
                    background: !jobId || submitting ? '#bdbdbd' : '#1565c0',
                    color: '#fff',
                    border: 'none',
                    borderRadius: 3,
                    cursor: !jobId || submitting ? 'not-allowed' : 'pointer',
                    fontSize: 13,
                    alignSelf: 'flex-start',
                  }}
                >
                  {submitting ? 'Submitting…' : 'Log correction'}
                </button>
              </form>

              {submitResult && (
                <div style={{ marginTop: 10, padding: '8px 12px', background: '#e8f5e9', border: '1px solid #81c784', borderRadius: 3, fontSize: 12, color: '#1b5e20' }}>
                  {submitResult}
                </div>
              )}
              {submitError && (
                <div style={{ marginTop: 10, padding: '8px 12px', background: '#ffe0e0', border: '1px solid #ef9a9a', borderRadius: 3, fontSize: 12, color: '#7f0000' }}>
                  {submitError}
                </div>
              )}
            </div>
          </>
        )}

        {!loadingDetail && !selected && plots.length > 0 && (
          <div style={{ color: '#9e9e9e', fontSize: 13, paddingTop: 20 }}>
            Select a plot from the list to review it.
          </div>
        )}
      </div>
    </div>
  );
}

const labelStyle: React.CSSProperties = { display: 'block', fontSize: 12, color: '#555', marginBottom: 3 };
const inputStyle: React.CSSProperties = { width: '100%', padding: '6px 8px', border: '1px solid #ccc', borderRadius: 3, fontSize: 13, fontFamily: 'monospace' };
