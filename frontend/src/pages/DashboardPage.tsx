import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
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
import {
  api,
  HealthResponse,
  RecentPredictionResponse,
  RunSummary,
  TrainStatus,
} from '../lib/api';
import { MetricCard } from '../components/MetricCard';

function fmt(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

function formatTs(ts?: string): string {
  if (!ts) return '—';
  if (!/^\d{8}_\d{6}$/.test(ts)) return ts;
  return `${ts.slice(0, 4)}-${ts.slice(4, 6)}-${ts.slice(6, 8)} ${ts.slice(9, 11)}:${ts.slice(11, 13)}`;
}

const RECENT_N = 30;

export function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latestRun, setLatestRun] = useState<RunSummary | null>(null);
  const [runsCount, setRunsCount] = useState(0);
  const [recent, setRecent] = useState<RecentPredictionResponse | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus>({ status: 'idle' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [h, runs, rec, ts] = await Promise.all([
        api.health(),
        api.runs(),
        api.recent(RECENT_N),
        api.trainStatus(),
      ]);
      setHealth(h);
      setRunsCount(runs.count);
      setLatestRun(runs.runs[0] ?? null);
      setRecent(rec);
      setTrainStatus(ts);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // 학습 중에는 4초마다 polling
  useEffect(() => {
    if (trainStatus.status !== 'running' && trainStatus.status !== 'pending') return;
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [trainStatus.status, refresh]);

  const trainingPct =
    trainStatus.last_epoch && trainStatus.total_epochs
      ? Math.round((trainStatus.last_epoch / trainStatus.total_epochs) * 100)
      : 0;

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <header className="mb-6 flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
          <p className="text-sm text-slate-500 mt-1">
            최신 모델·예측·학습 상태를 한눈에 확인합니다.
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          className="rounded-md bg-slate-900 text-white px-4 py-1.5 text-sm hover:bg-slate-700 disabled:opacity-50"
        >
          {loading ? '로딩…' : '새로고침'}
        </button>
      </header>

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-800">
          {error}
        </div>
      )}

      {/* 상단 상태 스트립 */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard
          label="모델 상태"
          value={health?.model_loaded ? '로드됨' : '미로드'}
          hint={health?.model_path?.split('/').slice(-1)[0]}
        />
        <MetricCard
          label="학습 상태"
          value={trainStatus.status}
          hint={
            trainStatus.last_epoch && trainStatus.total_epochs
              ? `${trainStatus.last_epoch}/${trainStatus.total_epochs} (${trainingPct}%)`
              : trainStatus.run_id ?? undefined
          }
        />
        <MetricCard label="총 Run 수" value={String(runsCount)} hint="results/" />
        <MetricCard
          label="특성 수"
          value={String(health?.n_features ?? '—')}
          hint={`seq=${health?.sequence_length ?? '—'}`}
        />
      </section>

      {(trainStatus.status === 'running' || trainStatus.status === 'pending') && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-1 text-xs text-slate-500">
            <span>Training in progress…</span>
            <span className="font-mono">{trainingPct}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
            <div
              className="h-full bg-slate-900 transition-all"
              style={{ width: `${trainingPct}%` }}
            />
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 최신 run 요약 */}
        <section className="lg:col-span-1 rounded-xl bg-white shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Latest Run</h2>
            <Link
              to="/history"
              className="text-xs text-slate-500 hover:text-slate-900 underline"
            >
              전체 보기
            </Link>
          </div>
          {latestRun ? (
            <>
              <div className="text-xs text-slate-500 mb-1">
                {formatTs(latestRun.timestamp)}
              </div>
              <div className="text-sm font-mono text-slate-900 mb-4">
                {latestRun.config_summary.model_type} · seq={latestRun.config_summary.sequence_length} ·
                lr={latestRun.config_summary.learning_rate}
              </div>
              <dl className="space-y-2 text-sm">
                <Row label="R²" value={fmt(latestRun.metrics.r2, 3)} />
                <Row label="RMSE" value={`${fmt(latestRun.metrics.rmse)} kW`} />
                <Row label="MAE" value={`${fmt(latestRun.metrics.mae)} kW`} />
                <Row
                  label="MAPE"
                  value={
                    latestRun.metrics.mape !== null
                      ? `${fmt(latestRun.metrics.mape, 2)}%`
                      : '—'
                  }
                />
                <Row
                  label="Direction Acc."
                  value={
                    latestRun.metrics.direction_accuracy !== null
                      ? `${fmt(latestRun.metrics.direction_accuracy, 2)}%`
                      : '—'
                  }
                />
              </dl>
            </>
          ) : (
            <div className="text-sm text-slate-400">
              아직 학습된 run이 없습니다.{' '}
              <Link to="/train" className="text-slate-700 underline">
                학습 시작
              </Link>
            </div>
          )}
        </section>

        {/* 최근 예측 차트 */}
        <section className="lg:col-span-2 rounded-xl bg-white shadow-sm border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">
              Recent Predictions
              <span className="ml-2 text-xs font-normal text-slate-500">
                test 마지막 {recent?.count ?? RECENT_N}개
              </span>
            </h2>
            <Link
              to="/predict"
              className="text-xs text-slate-500 hover:text-slate-900 underline"
            >
              상세
            </Link>
          </div>

          <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
            <Mini label="RMSE" value={recent ? recent.metrics.rmse.toFixed(0) : '—'} />
            <Mini label="MAE" value={recent ? recent.metrics.mae.toFixed(0) : '—'} />
            <Mini label="R²" value={recent ? recent.metrics.r2.toFixed(3) : '—'} />
          </div>

          <div className="h-64">
            {recent && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={recent.points}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="index" tick={{ fontSize: 11 }} />
                  <YAxis
                    tick={{ fontSize: 11 }}
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
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                    name="예측"
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-slate-100 py-1">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-mono font-medium text-slate-900">{value}</dd>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-slate-50 border border-slate-100 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono font-semibold text-slate-900">{value}</div>
    </div>
  );
}
