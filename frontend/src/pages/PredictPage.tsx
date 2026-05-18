import { useCallback, useEffect, useState } from 'react';
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
import { api, HealthResponse, RecentPredictionResponse } from '../lib/api';
import { MetricCard } from '../components/MetricCard';

const N_DEFAULT = 50;

export function PredictPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [data, setData] = useState<RecentPredictionResponse | null>(null);
  const [n, setN] = useState(N_DEFAULT);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (count: number) => {
    setLoading(true);
    setError(null);
    try {
      const [h, r] = await Promise.all([api.health(), api.recent(count)]);
      setHealth(h);
      setData(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(n);
  }, [load, n]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <header className="mb-8 flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Solar Demand Prediction</h1>
          <p className="text-sm text-slate-500 mt-1">
            FastAPI backend + React frontend (MVP vertical slice)
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-slate-600">샘플 수</label>
          <select
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm"
          >
            {[20, 50, 100, 200].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <button
            onClick={() => load(n)}
            className="rounded-md bg-slate-900 text-white px-4 py-1.5 text-sm hover:bg-slate-700 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? '로딩…' : '새로고침'}
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-800">
          {error}
        </div>
      )}

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="모델 상태"
          value={health?.model_loaded ? '로드됨' : '미로드'}
          hint={health?.model_path?.split('/').slice(-1)[0]}
        />
        <MetricCard
          label="RMSE"
          value={data ? data.metrics.rmse.toFixed(1) : '—'}
          hint="kW"
        />
        <MetricCard
          label="MAE"
          value={data ? data.metrics.mae.toFixed(1) : '—'}
          hint="kW"
        />
        <MetricCard
          label="R²"
          value={data ? data.metrics.r2.toFixed(3) : '—'}
          hint={`test 마지막 ${data?.count ?? 0}개`}
        />
      </section>

      <section className="rounded-xl bg-white shadow-sm border border-slate-200 p-5">
        <h2 className="text-lg font-semibold mb-4">예측 vs 실제</h2>
        <div className="h-96">
          {data && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.points}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="index" tick={{ fontSize: 12 }} />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => `${(v / 1000).toFixed(1)}k`}
                />
                <Tooltip
                  formatter={(v: number) => `${v.toFixed(0)} kW`}
                  labelFormatter={(i) => `Step ${i}`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#0f172a"
                  strokeWidth={2}
                  dot={false}
                  name="실제"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#f97316"
                  strokeWidth={2}
                  dot={false}
                  name="예측"
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>
    </div>
  );
}
