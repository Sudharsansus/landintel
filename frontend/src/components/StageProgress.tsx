import type { Stage } from '../api/client';

const STAGES: Stage[] = ['intake', 'extract', 'georef', 'assemble', 'report', 'delivered'];

const STAGE_LABEL: Record<Stage, string> = {
  intake:    'Intake',
  extract:   'M1 Extract',
  georef:    'M2 Georef',
  assemble:  'M3 Assemble',
  report:    'M4 Report',
  delivered: 'Delivered',
};

interface Props {
  current: Stage;
}

export function StageProgress({ current }: Props) {
  const activeIdx = STAGES.indexOf(current);

  return (
    <div style={{ display: 'flex', gap: 0, overflowX: 'auto', paddingBottom: 4 }}>
      {STAGES.map((stage, idx) => {
        const done = idx < activeIdx;
        const active = idx === activeIdx;
        return (
          <div
            key={stage}
            data-active={active ? 'true' : undefined}
            style={{
              flex: 1,
              minWidth: 80,
              textAlign: 'center',
              padding: '6px 4px',
              fontSize: 11,
              fontWeight: active ? 700 : 400,
              background: active ? '#1565c0' : done ? '#bbdefb' : '#f5f5f5',
              color: active ? '#fff' : done ? '#0d47a1' : '#9e9e9e',
              borderRight: idx < STAGES.length - 1 ? '1px solid #e0e0e0' : undefined,
              borderTop: '2px solid ' + (active ? '#1565c0' : done ? '#64b5f6' : '#e0e0e0'),
            }}
          >
            {STAGE_LABEL[stage]}
          </div>
        );
      })}
    </div>
  );
}
