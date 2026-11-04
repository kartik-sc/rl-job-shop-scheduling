"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  RefreshCw,
  Server,
  ShieldCheck,
  Wrench,
} from "lucide-react";
import { EpisodeLog } from "@/types/eventLog";
import { EPISODE_PRESETS, MOCK_EPISODE_LOG } from "@/lib/mockEpisode";
import { getMachineRiskAtTime, parseEpisodeLog } from "@/lib/parseEpisode";
import GanttChart from "@/components/GanttChart";
import RiskHeatmap from "@/components/RiskHeatmap";
import PlaybackControls from "@/components/PlaybackControls";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const POLICY_OPTIONS = [
  {
    key: "full_ppo",
    label: "Full Risk-Aware PPO",
    badge: "Canonical Mock",
    desc: "Balances C-MAPSS risk & queue state (1 PM, 1 Breakdown · 110u)",
  },
  {
    key: "threshold_spt",
    label: "Threshold + SPT (τ=0.70)",
    badge: "Heuristic Baseline",
    desc: "Strict rule-based PM whenever risk ≥ 0.70 (4 PMs, 0 Breakdowns · 116u)",
  },
  {
    key: "blind_ppo",
    label: "Maintenance-Blind PPO",
    badge: "Ablation Baseline",
    desc: "Ignores risk telemetry; runs to failure (0 PMs, 2 Breakdowns · 132u)",
  },
] as const;

export default function DashboardPage() {
  const [activePolicyKey, setActivePolicyKey] = useState<string>("full_ppo");
  const [apiEpisodeLog, setApiEpisodeLog] =
    useState<EpisodeLog>(MOCK_EPISODE_LOG);
  const [apiStatus, setApiStatus] = useState<
    "connected" | "fallback" | "loading"
  >("loading");
  const [currentTime, setCurrentTime] = useState<number>(110);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1);
  const [selectedJob, setSelectedJob] = useState<number | null>(null);

  const episodeLog = useMemo(() => {
    if (activePolicyKey === "full_ppo") {
      return apiEpisodeLog;
    }
    return EPISODE_PRESETS[activePolicyKey] ?? apiEpisodeLog;
  }, [activePolicyKey, apiEpisodeLog]);

  const parsed = useMemo(() => parseEpisodeLog(episodeLog), [episodeLog]);

  const fetchMockFromApi = useCallback(async () => {
    setApiStatus("loading");
    try {
      const res = await fetch(`${API_BASE_URL}/episodes/mock`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as EpisodeLog;
      setApiEpisodeLog(data);
      const parsedData = parseEpisodeLog(data);
      setCurrentTime(parsedData.makespan);
      setApiStatus("connected");
    } catch {
      setApiEpisodeLog(MOCK_EPISODE_LOG);
      setApiStatus("fallback");
    }
  }, []);

  useEffect(() => {
    fetchMockFromApi();
  }, [fetchMockFromApi]);

  const handleSelectPolicy = (key: string) => {
    setIsPlaying(false);
    setActivePolicyKey(key);
    const targetLog =
      key === "full_ppo"
        ? apiEpisodeLog
        : EPISODE_PRESETS[key] ?? apiEpisodeLog;
    const p = parseEpisodeLog(targetLog);
    setCurrentTime(p.makespan);
  };

  // Smooth playback loop
  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = 40;
    const stepPerTick = (speed * 0.45 * intervalMs) / 40;

    const timer = window.setInterval(() => {
      setCurrentTime((prev) => {
        if (prev >= parsed.makespan) {
          setIsPlaying(false);
          return parsed.makespan;
        }
        return Math.min(parsed.makespan, prev + stepPerTick);
      });
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [isPlaying, speed, parsed.makespan]);

  const handleTogglePlay = () => {
    if (!isPlaying && currentTime >= parsed.makespan) {
      setCurrentTime(0);
    }
    setIsPlaying((prev) => !prev);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleStep = (delta: number) => {
    setIsPlaying(false);
    setCurrentTime((prev) =>
      Math.max(0, Math.min(parsed.makespan, Math.round(prev + delta)))
    );
  };

  const handleScrub = useCallback(
    (newTime: number) => {
      setCurrentTime(Math.max(0, Math.min(parsed.makespan, newTime)));
    },
    [parsed.makespan]
  );

  // Live metrics at `currentTime`
  const completedOpsAtTime = useMemo(
    () =>
      parsed.operations.filter(
        (op) => !op.interrupted && op.end <= currentTime
      ).length,
    [parsed.operations, currentTime]
  );

  const totalCompletedOps = useMemo(
    () => parsed.operations.filter((op) => !op.interrupted).length,
    [parsed.operations]
  );

  const maxRiskAtTime = useMemo(() => {
    let maxR = 0;
    let maxMachine = 0;
    for (let m = 0; m < parsed.meta.n_machines; m++) {
      const r = getMachineRiskAtTime(parsed, m, currentTime);
      if (r > maxR) {
        maxR = r;
        maxMachine = m;
      }
    }
    return { risk: maxR, machine: maxMachine };
  }, [parsed, currentTime]);

  const totalPmDowntime = useMemo(
    () =>
      parsed.maintenances.reduce((acc, pm) => acc + (pm.end - pm.start), 0),
    [parsed.maintenances]
  );

  const totalBreakdownDowntime = useMemo(
    () =>
      parsed.breakdowns.reduce(
        (acc, bd) => acc + (bd.repairEndTime - bd.breakdownTime),
        0
      ),
    [parsed.breakdowns]
  );

  const recentEvents = useMemo(() => {
    return episodeLog.events
      .filter((ev) => ev.t <= currentTime)
      .slice(-10)
      .reverse();
  }, [episodeLog.events, currentTime]);

  return (
    <main className="mx-auto flex max-w-[1440px] flex-col gap-5 px-4 py-6 sm:px-6 lg:px-8">
      {/* Top Mission Control Header */}
      <header className="rajs-panel relative overflow-hidden rounded-2xl p-5 sm:px-6">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent" />

        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-cyan-400/40 bg-gradient-to-br from-cyan-500/25 to-blue-600/20 shadow-[0_0_25px_rgba(56,189,248,0.2)]">
              <ShieldCheck className="h-6 w-6 text-cyan-300" />
            </div>

            <div>
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="rounded-md border border-cyan-400/35 bg-cyan-500/15 px-2 py-0.5 font-mono text-[10px] font-bold tracking-widest text-cyan-300 uppercase">
                  RAJS · AIML EL
                </span>
                <h1 className="font-display text-xl font-bold tracking-tight text-white sm:text-2xl">
                  Risk-Aware Job-Shop Scheduling Telemetry
                </h1>
              </div>

              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-slate-400">
                <span>
                  Instance:{" "}
                  <strong className="text-slate-200">
                    {parsed.meta.instance}
                  </strong>
                </span>
                <span className="text-slate-700">•</span>
                <span>
                  Topology:{" "}
                  <strong className="text-slate-200">
                    {parsed.meta.n_machines}M × {parsed.meta.n_jobs}J (20 Ops)
                  </strong>
                </span>
                <span className="text-slate-700">•</span>
                <span>
                  Hazard Regime:{" "}
                  <strong className="text-amber-300 uppercase">
                    {parsed.meta.regime}
                  </strong>
                </span>
                <span className="text-slate-700">•</span>
                <span>
                  Seed: <strong className="text-slate-200">{parsed.meta.seed}</strong>
                </span>
              </div>
            </div>
          </div>

          {/* Right: Policy Selector & API Connection Status */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Policy Switcher */}
            <div className="flex items-center rounded-xl border border-slate-800 bg-slate-950/90 p-1">
              {POLICY_OPTIONS.map((opt) => {
                const active = activePolicyKey === opt.key;
                return (
                  <button
                    key={opt.key}
                    onClick={() => handleSelectPolicy(opt.key)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                      active
                        ? "bg-gradient-to-r from-cyan-500/25 to-blue-600/25 text-cyan-200 border border-cyan-400/40 shadow-sm"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>

            {/* FastAPI Status Badge */}
            <div
              className={`flex items-center gap-2 rounded-xl border px-3 py-2 font-mono text-xs font-medium ${
                apiStatus === "connected"
                  ? "border-emerald-500/40 bg-emerald-950/40 text-emerald-300"
                  : apiStatus === "loading"
                  ? "border-cyan-500/40 bg-cyan-950/40 text-cyan-300"
                  : "border-amber-500/40 bg-amber-950/40 text-amber-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "connected"
                    ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                    : "bg-amber-400"
                }`}
              />
              <Server className="h-3.5 w-3.5" />
              <span>
                {apiStatus === "connected"
                  ? "FastAPI :8000 Live"
                  : apiStatus === "loading"
                  ? "Syncing API..."
                  : "Local Mock Fallback"}
              </span>
            </div>

            <button
              onClick={fetchMockFromApi}
              title="Re-fetch /episodes/mock from FastAPI"
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-700/80 bg-slate-900 text-slate-300 transition hover:border-cyan-400/50 hover:text-white"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* 5 High-Contrast KPI Telemetry Cards */}
      <section className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-5">
        <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
          <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-cyan-400 to-blue-500" />
          <div className="flex items-center justify-between text-xs font-medium text-slate-400">
            <span>EPISODE MAKESPAN</span>
            <Clock className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-1.5 font-mono">
            <span className="text-2xl font-bold text-white">
              {parsed.makespan}u
            </span>
            <span className="text-xs text-cyan-400">
              (t = {currentTime.toFixed(0)}u)
            </span>
          </div>
          <div className="mt-1.5 text-[11px] text-slate-400">
            Policy:{" "}
            <span className="font-mono text-slate-200">{parsed.meta.policy}</span>
          </div>
        </div>

        <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
          <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-indigo-400 to-violet-500" />
          <div className="flex items-center justify-between text-xs font-medium text-slate-400">
            <span>OPERATIONS COMPLETED</span>
            <CheckCircle2 className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-1.5 font-mono">
            <span className="text-2xl font-bold text-white">
              {completedOpsAtTime}
            </span>
            <span className="text-sm text-slate-400">/ {totalCompletedOps}</span>
          </div>
          <div className="mt-1.5 text-[11px] text-slate-400">
            {Math.round(
              (completedOpsAtTime / Math.max(1, totalCompletedOps)) * 100
            )}
            % of shop schedule complete
          </div>
        </div>

        <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
          <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-amber-400 to-orange-500" />
          <div className="flex items-center justify-between text-xs font-medium text-slate-400">
            <span>PEAK SHOP RISK @ T={currentTime.toFixed(0)}</span>
            <Activity className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-amber-300">
              {(maxRiskAtTime.risk * 100).toFixed(0)}%
            </span>
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-xs text-amber-200">
              M0{maxRiskAtTime.machine}
            </span>
          </div>
          <div className="mt-1.5 text-[11px] text-slate-400">
            C-MAPSS Isotonic Risk ∈ [0, 1]
          </div>
        </div>

        <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
          <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-emerald-400 to-teal-500" />
          <div className="flex items-center justify-between text-xs font-medium text-slate-400">
            <span>PREVENTIVE MAINT.</span>
            <Wrench className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-emerald-300">
              {parsed.maintenances.length}
            </span>
            <span className="text-xs text-slate-400">
              ({totalPmDowntime}u total t_pm)
            </span>
          </div>
          <div className="mt-1.5 text-[11px] text-slate-400">
            {parsed.maintenances.length > 0
              ? `M0${parsed.maintenances[0].machine} reset to healthy (~0.08)`
              : "No preventive maintenance scheduled"}
          </div>
        </div>

        <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
          <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-rose-500 to-red-600" />
          <div className="flex items-center justify-between text-xs font-medium text-slate-400">
            <span>UNPLANNED BREAKDOWNS</span>
            <AlertTriangle className="h-4 w-4 text-rose-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-rose-300">
              {parsed.breakdowns.length}
            </span>
            <span className="text-xs text-slate-400">
              ({totalBreakdownDowntime}u t_repair)
            </span>
          </div>
          <div className="mt-1.5 text-[11px] text-slate-400">
            {parsed.breakdowns.length > 0
              ? `Cost: 5× t_pm + interrupted op restart`
              : "Zero breakdowns experienced"}
          </div>
        </div>
      </section>

      {/* Interactive Timeline Transport & Scrubber Deck */}
      <PlaybackControls
        episode={parsed}
        currentTime={currentTime}
        makespan={parsed.makespan}
        isPlaying={isPlaying}
        speed={speed}
        onTogglePlay={handleTogglePlay}
        onReset={handleReset}
        onStep={handleStep}
        onScrub={handleScrub}
        onChangeSpeed={setSpeed}
      />

      {/* Synchronized D3 Gantt Chart */}
      <GanttChart
        episode={parsed}
        currentTime={currentTime}
        onScrub={handleScrub}
        selectedJob={selectedJob}
        onSelectJob={setSelectedJob}
      />

      {/* Synchronized D3 Risk Heatmap & Live Engine Gauges */}
      <RiskHeatmap
        episode={parsed}
        currentTime={currentTime}
        onScrub={handleScrub}
      />

      {/* Bottom 2-Column Analytical Footer: Policy Comparison & Event Log Stream */}
      <section className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* Left: Policy Benchmark Comparison Table */}
        <div className="rajs-panel rounded-2xl p-5 lg:col-span-6">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-cyan-400" />
              <h2 className="font-display text-sm font-bold text-white sm:text-base">
                Policy Comparison on Instance ({parsed.meta.instance})
              </h2>
            </div>
            <span className="font-mono text-[11px] text-slate-400">
              Click row to inspect schedule
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 font-mono text-[10px] uppercase tracking-wider text-slate-400">
                  <th className="pb-2.5 pr-3">Policy</th>
                  <th className="pb-2.5 px-2">Makespan</th>
                  <th className="pb-2.5 px-2">PMs (t_pm)</th>
                  <th className="pb-2.5 px-2">Breakdowns</th>
                  <th className="pb-2.5 pl-2">Downtime</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {[
                  {
                    key: "full_ppo",
                    name: "Full Risk-Aware PPO",
                    tag: "Ours",
                    makespan: 110,
                    pms: 1,
                    breakdowns: 1,
                    downtime: 48,
                  },
                  {
                    key: "threshold_spt",
                    name: "Threshold + SPT (τ=0.70)",
                    tag: "Rule",
                    makespan: 116,
                    pms: 4,
                    breakdowns: 0,
                    downtime: 32,
                  },
                  {
                    key: "blind_ppo",
                    name: "Maintenance-Blind PPO",
                    tag: "Blind",
                    makespan: 132,
                    pms: 0,
                    breakdowns: 2,
                    downtime: 80,
                  },
                ].map((row) => {
                  const isCurrent = activePolicyKey === row.key;
                  return (
                    <tr
                      key={row.key}
                      onClick={() => handleSelectPolicy(row.key)}
                      className={`cursor-pointer transition ${
                        isCurrent
                          ? "bg-cyan-500/10 text-white"
                          : "text-slate-300 hover:bg-slate-800/40"
                      }`}
                    >
                      <td className="py-3 pr-3 font-sans font-semibold">
                        <div className="flex items-center gap-2">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              isCurrent ? "bg-cyan-400" : "bg-slate-600"
                            }`}
                          />
                          <span>{row.name}</span>
                          <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
                            {row.tag}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-2 font-bold text-cyan-300">
                        {row.makespan}u
                      </td>
                      <td className="py-3 px-2 text-emerald-300">
                        {row.pms} ({row.pms * 8}u)
                      </td>
                      <td className="py-3 px-2 text-rose-300">
                        {row.breakdowns} ({row.breakdowns * 40}u)
                      </td>
                      <td className="py-3 pl-2 text-slate-300">
                        {row.downtime}u
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Chronological Event Stream Inspector */}
        <div className="rajs-panel rounded-2xl p-5 lg:col-span-6">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              <h2 className="font-display text-sm font-bold text-white sm:text-base">
                Schema Event Log Stream (t ≤ {currentTime.toFixed(0)}u)
              </h2>
            </div>
            <span className="font-mono text-xs text-slate-400">
              {episodeLog.events.filter((e) => e.t <= currentTime).length} /{" "}
              {episodeLog.events.length} events
            </span>
          </div>

          <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {recentEvents.map((ev, idx) => (
              <div
                key={`${ev.t}-${ev.type}-${ev.machine ?? "all"}-${idx}`}
                className="rajs-card-inner flex items-center justify-between rounded-lg px-3 py-1.5 font-mono text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="font-bold text-cyan-400">t={ev.t}u</span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      ev.type === "breakdown"
                        ? "border border-rose-500/40 bg-rose-950/80 text-rose-300"
                        : ev.type.startsWith("maintenance")
                        ? "border border-emerald-500/40 bg-emerald-950/80 text-emerald-300"
                        : ev.type === "risk_update"
                        ? "bg-amber-950/60 text-amber-300"
                        : "bg-slate-800/90 text-slate-200"
                    }`}
                  >
                    {ev.type}
                  </span>
                </div>
                <div className="text-[11px] text-slate-300">
                  {ev.machine !== undefined && `M0${ev.machine} `}
                  {ev.job !== undefined && `J${ev.job}·O${ev.op} `}
                  {ev.risk !== undefined && `r=${ev.risk.toFixed(2)}`}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
