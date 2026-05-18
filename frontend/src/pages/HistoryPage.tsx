import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api, RunDetail, RunSummary } from '../lib/api';
import { MetricCard } from '../components/MetricCard';

function formatTs(ts: string): string {
  // "20260518_101115" → "2026-05-18 10:11:15"
  if (!/^\d{8}_\d{6}$/.test(ts)) return ts;
  return `${ts.slice(0, 4)}-${ts.slice(4, 6)}-${ts.slice(6, 8)} ${ts.slice(9, 11)}:${ts.slice(11, 13)}:${ts.slice(13, 15)}`;
}

function fmt(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

export function HistoryPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api
      .runs()
      .then((r) => {
        setRuns(r.runs);
        if (r.runs.length > 0) setSelectedId(r.runs[0].run_id);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    api
      .run(selectedId)
      .then(setDetail)
      .catch((e) => setError(String(e)));
  }, [selectedId]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-slate-900">Training History</h1>
        <p className="text-sm text-slate-500 mt-1">backend/results/ 파싱 결과</p>
      </header>

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-800">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <aside className="lg:col-span-1 space-y-2">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
            Runs ({runs.length})
          </div>
          {loading && <div className="text-sm text-slate-400">로딩…</div>}
          {runs.map((r) => {
            const active = r.run_id === selectedId;
            return (
              <button
                key={r.run_id}
                onClick={() => setSelectedId(r.run_id)}
                className={[
                  'w-full text-left rounded-lg border px-4 py-3 transition',
                  active
                    ? 'border-slate-900 bg-slate-900 text-white'
                    : 'border-slate-200 bg-white hover:border-slate-400',
                ].join(' ')}
              >
                <div className="text-sm font-medium">{formatTs(r.timestamp)}</div>
                <div className={`mt-1 text-xs ${active ? 'text-slate-300' : 'text-slate-500'}`}>
                  {r.config_summary.model_type} · R² {fmt(r.metrics.r2, 3)}
                </div>
              </button>
            );
          })}
        </aside>

        <section className="lg:col-span-2 space-y-6">
          {detail ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard label="RMSE" value={fmt(detail.metrics.rmse)} hint="kW" />
                <MetricCard label="MAE" value={fmt(detail.metrics.mae)} hint="kW" />
                <MetricCard label="R²" value={fmt(detail.metrics.r2, 3)} />
                <MetricCard
                  label="MAPE"
                  value={detail.metrics.mape !== null ? `${fmt(detail.metrics.mape, 2)}%` : '—'}
                />
              </div>

              <div className="rounded-xl bg-white shadow-sm border border-slate-200 p-5">
                <h2 className="text-lg font-semibold mb-1">Training Loss</h2>
                <p className="text-xs text-slate-500 mb-3">
                  {detail.history.length} epochs · {detail.config_summary.loss} loss
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={detail.history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#0f172a"
                        dot={false}
                        name="train loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="val_loss"
                        stroke="#f97316"
                        dot={false}
                        name="val loss"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-xl bg-white shadow-sm border border-slate-200 p-5">
                <h2 className="text-lg font-semibold mb-3">Config</h2>
                <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                  {Object.entries(detail.config_summary).map(([k, v]) => (
                    <div key={k} className="flex justify-between border-b border-slate-100 py-1">
                      <dt className="text-slate-500">{k}</dt>
                      <dd className="font-mono text-slate-900">{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-300 px-6 py-10 text-center text-slate-500">
              왼쪽에서 run을 선택하세요
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
