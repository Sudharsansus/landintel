import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { StageProgress } from './StageProgress';

describe('StageProgress', () => {
  it('renders all six stage labels', () => {
    render(<StageProgress current="intake" />);
    expect(screen.getByText('Intake')).toBeInTheDocument();
    expect(screen.getByText('M1 Extract')).toBeInTheDocument();
    expect(screen.getByText('M2 Georef')).toBeInTheDocument();
    expect(screen.getByText('M3 Assemble')).toBeInTheDocument();
    expect(screen.getByText('M4 Report')).toBeInTheDocument();
    expect(screen.getByText('Delivered')).toBeInTheDocument();
  });

  it('highlights intake stage as active', () => {
    const { container } = render(<StageProgress current="intake" />);
    const activeEl = container.querySelector('[data-active="true"]') as HTMLElement | null;
    expect(activeEl).not.toBeNull();
    expect(activeEl?.textContent).toBe('Intake');
  });

  it('highlights delivered stage when delivered', () => {
    const { container } = render(<StageProgress current="delivered" />);
    const activeEl = container.querySelector('[data-active="true"]') as HTMLElement | null;
    expect(activeEl?.textContent).toBe('Delivered');
  });

  it('highlights extract when stage is extract', () => {
    const { container } = render(<StageProgress current="extract" />);
    const activeEl = container.querySelector('[data-active="true"]') as HTMLElement | null;
    expect(activeEl?.textContent).toBe('M1 Extract');
  });
});
