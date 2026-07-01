import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { PlotDetail } from '../api/client';
import { PlotViewer } from './PlotViewer';

function makePlot(overrides: Partial<PlotDetail> = {}): PlotDetail {
  return {
    survey_no: '100',
    status: 'flagged',
    stated_area: 1.665,
    flags: ['area mismatch: 5.1% > 5.0% tolerance'],
    district: 'Sivagangai',
    taluk: 'Manamadurai',
    village: 'T.Pudukkottai',
    scale: 2021,
    measurement_count: 12,
    corner_count: 27,
    boundary_closed: true,
    ...overrides,
  };
}

describe('PlotViewer', () => {
  it('shows survey number', () => {
    render(<PlotViewer plot={makePlot()} />);
    expect(screen.getByText(/Survey 100/)).toBeInTheDocument();
  });

  it('shows district, taluk, village', () => {
    render(<PlotViewer plot={makePlot()} />);
    expect(screen.getByText('Sivagangai')).toBeInTheDocument();
    expect(screen.getByText('Manamadurai')).toBeInTheDocument();
    expect(screen.getByText('T.Pudukkottai')).toBeInTheDocument();
  });

  it('shows stated area with unit', () => {
    render(<PlotViewer plot={makePlot()} />);
    expect(screen.getByText('1.665 ha')).toBeInTheDocument();
  });

  it('shows null stated area as dash', () => {
    render(<PlotViewer plot={makePlot({ stated_area: null })} />);
    const cells = screen.getAllByText('—');
    expect(cells.length).toBeGreaterThan(0);
  });

  it('shows drawing scale', () => {
    render(<PlotViewer plot={makePlot()} />);
    expect(screen.getByText('1 : 2021')).toBeInTheDocument();
  });

  it('shows Closed for closed boundary', () => {
    render(<PlotViewer plot={makePlot({ boundary_closed: true })} />);
    expect(screen.getByText('Closed')).toBeInTheDocument();
  });

  it('shows Open for non-closing boundary', () => {
    render(<PlotViewer plot={makePlot({ boundary_closed: false })} />);
    expect(screen.getByText('Open (anomaly)')).toBeInTheDocument();
  });

  it('shows flag text', () => {
    render(<PlotViewer plot={makePlot()} />);
    expect(screen.getByText(/area mismatch/)).toBeInTheDocument();
  });

  it('shows no flags section when flags list is empty', () => {
    render(<PlotViewer plot={makePlot({ flags: [] })} />);
    expect(screen.queryByText(/Flags/)).toBeNull();
  });

  it('shows corner stone count', () => {
    render(<PlotViewer plot={makePlot()} />);
    expect(screen.getByText('27')).toBeInTheDocument();
  });
});
