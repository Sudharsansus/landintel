import type { PlotDetail } from '../api/client';

// PlotViewer shows the structured data extracted for a flagged plot.
// It is NOT a DXF canvas renderer — that would require a full CAD library.
// The boundary geometry lives in the DXF output file; this component surfaces
// the numbers the reviewer needs to make a correction decision.

interface Props {
  plot: PlotDetail;
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <tr>
      <td style={{ padding: '4px 8px', color: '#666', width: 160, verticalAlign: 'top' }}>{label}</td>
      <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 13 }}>{value}</td>
    </tr>
  );
}

export function PlotViewer({ plot }: Props) {
  const closureText =
    plot.boundary_closed === null ? '—' :
    plot.boundary_closed ? 'Closed' : 'Open (anomaly)';

  const closureColor =
    plot.boundary_closed === null ? '#666' :
    plot.boundary_closed ? '#1b5e20' : '#c62828';

  return (
    <div style={{ border: '1px solid #d0d0d0', borderRadius: 3, background: '#fff', padding: 16 }}>
      <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 12 }}>
        Survey {plot.survey_no}
        <span style={{
          marginLeft: 10,
          fontSize: 11,
          fontWeight: 600,
          padding: '2px 8px',
          borderRadius: 3,
          background: plot.status === 'flagged' ? '#fff2cc' : '#e8f5e9',
          color: plot.status === 'flagged' ? '#7a5900' : '#1b5e20',
          border: '1px solid ' + (plot.status === 'flagged' ? '#e6c62b' : '#81c784'),
        }}>
          {plot.status}
        </span>
      </div>

      <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 12 }}>
        <tbody>
          <Row label="District" value={plot.district} />
          <Row label="Taluk" value={plot.taluk} />
          <Row label="Village" value={plot.village} />
          <Row label="Stated area" value={plot.stated_area != null ? `${plot.stated_area} ha` : '—'} />
          <Row label="Drawing scale" value={plot.scale != null ? `1 : ${plot.scale}` : '—'} />
          <Row label="Boundary" value={<span style={{ color: closureColor }}>{closureText}</span>} />
          <Row label="Corner stones" value={plot.corner_count} />
          <Row label="Measurements" value={plot.measurement_count} />
        </tbody>
      </table>

      {plot.flags.length > 0 && (
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#7a5900', marginBottom: 4 }}>
            Flags ({plot.flags.length})
          </div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#555' }}>
            {plot.flags.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
