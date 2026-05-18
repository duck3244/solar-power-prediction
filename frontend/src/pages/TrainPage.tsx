import { useCallback, useEffect, useRef, useState } from 'react';
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
import { api, TrainRequest, TrainStatus } from '../lib/api';
import { MetricCard } from '../components/MetricCard';

type EpochPoint = {
  epoch: number;
  loss: number;
  val_loss?: number | null;
};

const DEFAULT_CONFIG: TrainRequest = {
  epochs: 5,
  batch_size: 32,
  learning_rate: 0.001,
  model_type: 'advanced',
  early_stopping_patience: 5,
};

export function TrainPage() {
  const [config, setConfig] = useState<TrainRequest>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<TrainStatus>({ status: 'idle' });
  const [history, setHistory] = useState<EpochPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [doneInfo, setDoneInfo] = useState<Record<string, number | null> | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  const closeStream = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, []);

  useEffect(() => () => closeStream(), [closeStream]);

  // 페이지 진입 시 진행 중 학습이 있으면 상태 조회
  useEffect(() => {
    api
      .trainStatus()
      .then((s) => {
        if (s.status === 'running' || s.status === 'pending') {
          setStatus(s);
          if (s.run_id) subscribe(s.run_id);
        }
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function subscribe(runId: string) {
    closeStream();
    const es = new EventSource(api.trainStreamUrl(runId));
    sourceRef.current = es;

    es.addEventListener('start', (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setStatus((s) => ({ ...s, status: 'running', run_id: data.run_id }));
    });

    es.addEventListener('epoch', (ev) => {
      const d = JSON.parse((ev as MessageEvent).data);
      setStatus((s) => ({
        ...s,
        status: 'running',
        last_epoch: d.epoch,
        total_epochs: d.total,
      }));
      setHistory((prev) => [
        ...prev,
        { epoch: d.epoch, loss: d.loss, val_loss: d.val_loss ?? null },
      ]);
    });

    es.addEventListener('done', (ev) => {
      const d = JSON.parse((ev as MessageEvent).data);
      setDoneInfo(d.metrics ?? null);
      setStatus((s) => ({ ...s, status: 'completed', metrics: d.metrics }));
      closeStream();
    });

    es.addEventListener('error', (ev) => {
      const data = (ev as MessageEvent).data;
      try {
        const d = data ? JSON.parse(data) : null;
        setError(d?.message ?? '학습 실패');
      } catch {
        setError('SSE 연결 오류');
      }
      setStatus((s) => ({ ...s, status: 'failed' }));
      closeStream();
    });
  }

  async function onStart() {
    setError(null);
    setHistory([]);
    setDoneInfo(null);
    try {
      const s = await api.startTrain(config);
      setStatus(s);
      if (s.run_id) subscribe(s.run_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const running = status.status === 'running' || status.status === 'pending';
  const progress =
    status.last_epoch && status.total_epochs
      ? Math.round((status.last_epoch / status.total_epochs) * 100)
      : 0;

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-slate-900">Train</h1>
        <p className="text-sm text-slate-500 mt-1">
          학습을 트리거하고 SSE로 epoch 진행률을 실시간 수신합니다.
        </p>
      </header>

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-800">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <aside className="lg:col-span-1 rounded-xl bg-white border border-slate-200 p-5 space-y-3">
          <h2 className="text-base font-semibold">Config</h2>

          <NumField
            label="epochs"
            value={config.epochs}
            onChange={(v) => setConfig({ ...config, epochs: v })}
            min={1}
            max={500}
            disabled={running}
          />
          <NumField
            label="batch_size"
            value={config.batch_size}
            onChange={(v) => setConfig({ ...config, batch_size: v })}
            min={1}
            max={512}
            disabled={running}
          />
          <NumField
            label="learning_rate"
            value={config.learning_rate}
            onChange={(v) => setConfig({ ...config, learning_rate: v })}
            min={0.00001}
            max={1}
            step={0.0001}
            disabled={running}
          />
          <NumField
            label="early_stopping_patience"
            value={config.early_stopping_patience}
            onChange={(v) => setConfig({ ...config, early_stopping_patience: v })}
            min={1}
            max={100}
            disabled={running}
          />
          <SelectField
            label="model_type"
            value={config.model_type ?? 'advanced'}
            options={['basic', 'advanced', 'transformer']}
            onChange={(v) =>
              setConfig({ ...config, model_type: v as TrainRequest['model_type'] })
            }
            disabled={running}
          />

          <button
            onClick={onStart}
            disabled={running}
            className="w-full rounded-md bg-slate-900 text-white py-2 text-sm font-medium hover:bg-slate-700 disabled:opacity-50"
          >
            {running ? '학습 중…' : 'Start Training'}
          </button>
        </aside>

        <section className="lg:col-span-2 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard
              label="Status"
              value={status.status}
              hint={status.run_id ?? ''}
            />
            <MetricCard
              label="Epoch"
              value={
                status.last_epoch && status.total_epochs
                  ? `${status.last_epoch}/${status.total_epochs}`
                  : '—'
              }
              hint={running ? `${progress}%` : undefined}
            />
            <MetricCard
              label="Current Loss"
              value={
                history.length > 0 ? history[history.length - 1].loss.toFixed(4) : '—'
              }
            />
            <MetricCard
              label="Current Val Loss"
              value={
                history.length > 0 && history[history.length - 1].val_loss != null
                  ? (history[history.length - 1].val_loss as number).toFixed(4)
                  : '—'
              }
            />
          </div>

          {running && (
            <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
              <div
                className="h-full bg-slate-900 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          <div className="rounded-xl bg-white shadow-sm border border-slate-200 p-5">
            <h2 className="text-lg font-semibold mb-3">Live Loss</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
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
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    stroke="#f97316"
                    dot={false}
                    name="val loss"
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {doneInfo && (
            <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-4 text-sm text-emerald-900">
              <div className="font-semibold mb-1">학습 완료 — 모델이 자동 교체되었습니다</div>
              <div className="font-mono text-xs">
                {Object.entries(doneInfo)
                  .map(([k, v]) => `${k}: ${v?.toFixed ? v.toFixed(4) : v}`)
                  .join('  •  ')}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

type NumFieldProps = {
  label: string;
  value?: number;
  onChange: (v: number | undefined) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
};

function NumField({ label, value, onChange, min, max, step = 1, disabled }: NumFieldProps) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
      <input
        type="number"
        value={value ?? ''}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === '' ? undefined : Number(v));
        }}
        className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm font-mono disabled:bg-slate-50"
      />
    </label>
  );
}

type SelectFieldProps = {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  disabled?: boolean;
};

function SelectField({ label, value, options, onChange, disabled }: SelectFieldProps) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm disabled:bg-slate-50"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
