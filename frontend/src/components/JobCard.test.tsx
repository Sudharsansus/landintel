import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { Job } from '../api/client';
import { JobCard } from './JobCard';

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: 'abc123def456789012345678',
    client_id: 'client1',
    status: 'queued',
    stage: 'intake',
    input_files: ['key1.pdf'],
    output_files: [],
    audit: [],
    created_at: '2026-06-05T10:00:00Z',
    ...overrides,
  };
}

describe('JobCard', () => {
  it('shows truncated job ID', () => {
    render(<JobCard job={makeJob()} onClick={() => {}} />);
    expect(screen.getByText(/abc123def456/)).toBeInTheDocument();
  });

  it('shows queued status badge', () => {
    render(<JobCard job={makeJob({ status: 'queued' })} onClick={() => {}} />);
    expect(screen.getByText('queued')).toBeInTheDocument();
  });

  it('shows needs review status', () => {
    render(<JobCard job={makeJob({ status: 'needs_review' })} onClick={() => {}} />);
    expect(screen.getByText('needs review')).toBeInTheDocument();
  });

  it('shows failed status', () => {
    render(<JobCard job={makeJob({ status: 'failed' })} onClick={() => {}} />);
    expect(screen.getByText('failed')).toBeInTheDocument();
  });

  it('shows file count', () => {
    render(<JobCard job={makeJob({ input_files: ['a.pdf', 'b.pdf'] })} onClick={() => {}} />);
    expect(screen.getByText(/2 files/)).toBeInTheDocument();
  });

  it('shows singular file', () => {
    render(<JobCard job={makeJob({ input_files: ['a.pdf'] })} onClick={() => {}} />);
    expect(screen.getByText(/1 file\b/)).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    let clicked = false;
    render(<JobCard job={makeJob()} onClick={() => { clicked = true; }} />);
    screen.getByText(/abc123/).parentElement!.click();
    expect(clicked).toBe(true);
  });
});
