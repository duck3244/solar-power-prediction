import { NavLink } from 'react-router-dom';

const linkClass = ({ isActive }: { isActive: boolean }) =>
  [
    'px-3 py-1.5 rounded-md text-sm font-medium transition',
    isActive
      ? 'bg-slate-900 text-white'
      : 'text-slate-600 hover:bg-slate-200 hover:text-slate-900',
  ].join(' ');

export function NavBar() {
  return (
    <nav className="border-b border-slate-200 bg-white">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="text-base font-semibold text-slate-900">☀ Solar Demand</div>
        <div className="flex items-center gap-1">
          <NavLink to="/dashboard" className={linkClass}>
            Dashboard
          </NavLink>
          <NavLink to="/predict" className={linkClass}>
            Predict
          </NavLink>
          <NavLink to="/train" className={linkClass}>
            Train
          </NavLink>
          <NavLink to="/history" className={linkClass}>
            History
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
