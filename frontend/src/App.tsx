import { Link, Route, Routes, useLocation } from 'react-router-dom';
import { JobDetail } from './pages/JobDetail';
import { Jobs } from './pages/Jobs';
import { Review } from './pages/Review';
import { Upload } from './pages/Upload';

const NAV_LINKS = [
  { to: '/upload', label: 'Upload' },
  { to: '/jobs', label: 'Jobs' },
  { to: '/review', label: 'Review' },
];

export function App() {
  const { pathname } = useLocation();

  return (
    <div style={{ minHeight: '100vh', background: '#fafafa' }}>
      <header style={{
        background: '#1a237e',
        color: '#fff',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        height: 50,
        gap: 32,
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
      }}>
        <Link to="/jobs" style={{ color: '#fff', textDecoration: 'none', fontWeight: 700, fontSize: 15, letterSpacing: '0.02em' }}>
          LandIntel
        </Link>
        <nav style={{ display: 'flex', gap: 8 }}>
          {NAV_LINKS.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              style={{
                color: pathname.startsWith(to) ? '#90caf9' : 'rgba(255,255,255,0.75)',
                textDecoration: 'none',
                fontSize: 13,
                padding: '4px 10px',
                borderRadius: 3,
                background: pathname.startsWith(to) ? 'rgba(255,255,255,0.1)' : 'transparent',
              }}
            >
              {label}
            </Link>
          ))}
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<Jobs />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/jobs/:id" element={<JobDetail />} />
          <Route path="/review" element={<Review />} />
        </Routes>
      </main>
    </div>
  );
}
