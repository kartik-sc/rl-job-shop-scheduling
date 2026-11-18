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
import { EpisodeLog, EpisodeSummary } from "@/types/eventLog";
import { MOCK_EPISODE_LOG } from "@/lib/mockEpisode";
import {
  fetchEpisodes,
  loadEpisode,
  localEpisodes,
  MOCK_LABEL,
} from "@/lib/episodes";
import { getMachineRiskAtTime, parseEpisodeLog } from "@/lib/parseEpisode";
import GanttChart from "@/components/GanttChart";
import RiskHeatmap from "@/components/RiskHeatmap";
import PlaybackControls from "@/components/PlaybackControls";
import FactoryFloor from "@/components/FactoryFloor";
import RLBrain from "@/components/RLBrain";
import TrainingArena, { RaceEntry } from "@/components/TrainingArena";
import Sandbox from "@/components/Sandbox";
import { MockBadge, MockNotice } from "@/components/MockBadge";
import { buildEpisodeScene, buildFlows, buildRoutes } from "@/lib/scene";
import { deriveDecisions, syntheticTraining } from "@/lib/rl";

type Tab = "overview" | "floor" | "rl" | "arena" | "sandbox";
const TABS: { key: Tab; label: string; emoji: string }[] = [
  { key: "overview", label: "Overview", emoji: "◰" },
  { key: "floor", label: "Factory Floor", emoji: "⚙" },
  { key: "rl", label: "RL Brain", emoji: "✦" },
  { key: "arena", label: "Training Arena", emoji: "⚑" },
  { key: "sandbox", label: "Sandbox", emoji: "▶" },
];

function useAnimatedNumber(target: number, ms = 600) {
  const [v, setV] = useState(target);
  const from = React.useRef(target);
  useEffect(() => {
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      const cur = a + (target - a) * e;
      from.current = cur;
      setV(cur);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

function Num({ value, digits = 0, suffix = "" }: { value: number; digits?: number; suffix?: string }) {
  const v = useAnimatedNumber(value);
  return (
    <>
      {v.toFixed(digits)}
      {suffix}
    </>
  );
}

const RACE_COLORS = ["#f0abfc", "#38bdf8", "#fb923c", "#34d399"];

/** Playback covers a whole episode in about this many seconds at 1x. */
const PLAYBACK_SECONDS = 10;

function formatDowntime(episode: EpisodeSummary): string {
  const { maintenance_downtime: pm, repair_downtime: repair } = episode;
  return pm === null || repair === null ? "—" : `${pm + repair}u`;
}

export default function DashboardPage() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>(() =>
    localEpisodes(false)
  );
  const [activeId, setActiveId] = useState<string>("local-mock-full_ppo");
  const [episodeLog, setEpisodeLog] = useState<EpisodeLog>(MOCK_EPISODE_LOG);
  const [apiStatus, setApiStatus] = useState<
    "connected" | "fallback" | "loading"
  >("loading");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1);
  const [selectedJob, setSelectedJob] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("floor");
  const [selectedMachine, setSelectedMachine] = useState<number | null>(null);
  const [comparisonLogs, setComparisonLogs] = useState<Record<string, EpisodeLog>>({});

  const parsed = useMemo(() => parseEpisodeLog(episodeLog), [episodeLog]);
  const routes = useMemo(() => buildRoutes(parsed), [parsed]);
  const flows = useMemo(() => buildFlows(routes), [routes]);
  const scene = buildEpisodeScene(parsed, routes, flows, currentTime);
  const decisions = useMemo(() => deriveDecisions(parsed, episodeLog), [parsed, episodeLog]);
  const curves = useMemo(() => syntheticTraining(), []);

  const activeEpisode = useMemo(
    () => episodes.find((e) => e.id === activeId),
    [episodes, activeId]
  );
  const isMock = activeEpisode?.source !== "real";

  /** The first episode autoplays from t=0; one picked later is shown whole. */
  const selectEpisode = useCallback(async (id: string, autoplay = false) => {
    setIsPlaying(false);
    setSelectedJob(null);
    setSelectedMachine(null);
    setLoadError(null);
    try {
      const log = await loadEpisode(id);
      setActiveId(id);
      setEpisodeLog(log);
      setCurrentTime(autoplay ? 0 : parseEpisodeLog(log).makespan);
      setIsPlaying(autoplay);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const refreshEpisodes = useCallback(async () => {
    setApiStatus("loading");
    try {
      const served = await fetchEpisodes();
      const all = [...served, ...localEpisodes(true)];
      setEpisodes(all);
      setApiStatus("connected");
      await selectEpisode(all[0].id, true);
    } catch {
      setEpisodes(localEpisodes(false));
      setApiStatus("fallback");
      await selectEpisode("local-mock-full_ppo", true);
    }
  }, [selectEpisode]);

  useEffect(() => {
    refreshEpisodes();
  }, [refreshEpisodes]);

  const comparisonRows = useMemo(
    () =>
      episodes.filter(
        (e) =>
          e.source === activeEpisode?.source &&
          e.meta.instance === activeEpisode.meta.instance
      ),
    [episodes, activeEpisode]
  );

  useEffect(() => {
    const missing = comparisonRows.filter((row) => !(row.id in comparisonLogs));
    if (missing.length === 0) return;
    let cancelled = false;
    Promise.all(missing.map(async (row) => [row.id, await loadEpisode(row.id)] as const))
      .then((loaded) => {
        if (!cancelled) setComparisonLogs((prev) => ({ ...prev, ...Object.fromEntries(loaded) }));
      })
      .catch((error) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [comparisonRows, comparisonLogs]);

  const raceEntries: RaceEntry[] = useMemo(
    () =>
      comparisonRows.flatMap((row, i) => {
        const log = comparisonLogs[row.id];
        if (!log) return [];
        const color = RACE_COLORS[i % RACE_COLORS.length];
        return [{ key: row.id, name: row.label, color, parsed: parseEpisodeLog(log) }];
      }),
    [comparisonRows, comparisonLogs]
  );

  // Smooth playback loop
  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = 40;
    const stepPerTick =
      (speed * parsed.makespan * intervalMs) / (PLAYBACK_SECONDS * 1000);

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
                <h1 className="font-display text-gradient text-xl font-bold tracking-tight sm:text-2xl">
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
                    {parsed.meta.n_machines}M × {parsed.meta.n_jobs}J (
                    {totalCompletedOps} Ops)
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

          {/* Right: Episode Selector & API Connection Status */}
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/90 px-3 py-1.5 text-xs text-slate-400">
              <span className="font-semibold">Episode</span>
              <select
                value={activeId}
                onChange={(e) => selectEpisode(e.target.value)}
                className="max-w-[22rem] rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 font-semibold text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                {(["real", "mock"] as const).map((source) => {
                  const group = episodes.filter((e) => e.source === source);
                  if (group.length === 0) return null;
                  return (
                    <optgroup
                      key={source}
                      label={source === "real" ? "Real health model" : MOCK_LABEL}
                    >
                      {group.map((e) => (
                        <option key={e.id} value={e.id}>
                          {e.label}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
            </label>

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
              onClick={refreshEpisodes}
              title="Re-fetch /episodes from FastAPI"
              aria-label="Re-fetch episodes from the API"
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-700/80 bg-slate-900 text-slate-300 transition hover:border-cyan-400/50 hover:text-white"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Data source: every mock episode says so, every real one names its policy */}
      {isMock ? (
        <div
          role="status"
          className="flex flex-wrap items-center gap-3 rounded-2xl border-2 border-amber-400/70 bg-amber-950/60 px-5 py-3"
        >
          <MockBadge size="md" />
          <span className="text-sm font-semibold text-amber-100">
            {activeEpisode?.label ?? `${MOCK_LABEL}: hand-written episode`}
          </span>
          <span className="text-xs text-amber-200/80">
            Hand-written for UI development. Not a simulation result.
          </span>
        </div>
      ) : (
        <div
          role="status"
          className="flex flex-wrap items-center gap-3 rounded-2xl border border-emerald-500/40 bg-emerald-950/30 px-5 py-3"
        >
          <span className="text-sm font-bold text-emerald-200">
            {activeEpisode?.label}
          </span>
          <span className="font-mono text-xs text-emerald-300/80">
            {activeEpisode?.detail}
          </span>
          {parsed.meta.health_model && (
            <span className="font-mono text-xs text-slate-400">
              health model: {parsed.meta.health_model} ({parsed.meta.pool}{" "}
              engines)
            </span>
          )}
        </div>
      )}
      {loadError && (
        <div
          role="alert"
          className="rounded-xl border border-rose-500/50 bg-rose-950/50 px-4 py-2 font-mono text-xs text-rose-200"
        >
          Could not load episode: {loadError}
        </div>
      )}
      {/* Animated tab navigation */}
      <nav className="rajs-panel relative flex flex-wrap gap-1 rounded-2xl p-1.5">
        {TABS.map((tb) => {
          const active = tab === tb.key;
          return (
            <button
              key={tb.key}
              onClick={() => {
                setTab(tb.key);
                if (tb.key === "sandbox") setIsPlaying(false);
              }}
              className={`relative flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all duration-300 ${
                active
                  ? "bg-gradient-to-r from-cyan-500/25 via-indigo-500/25 to-fuchsia-500/25 text-white shadow-[0_0_24px_-6px_rgba(129,140,248,0.7)] ring-1 ring-white/15"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
              }`}
            >
              <span className={`text-base ${active ? "text-cyan-300" : ""}`}>{tb.emoji}</span>
              {tb.label}
            </button>
          );
        })}
      </nav>

      {/* 5 High-Contrast KPI Telemetry Cards */}
      {tab !== "sandbox" && (
      <section key={`kpi-${tab}`} className="anim-fade-up grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-5">
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
              <Num value={completedOpsAtTime} />
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
              <Num value={maxRiskAtTime.risk * 100} suffix="%" />
            </span>
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-xs text-amber-200">
              M{maxRiskAtTime.machine}
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
              ? `First on M${parsed.maintenances[0].machine} at t=${parsed.maintenances[0].start}u`
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
              ? "Interrupted operations restart after repair"
              : "Zero breakdowns experienced"}
          </div>
        </div>
      </section>
      )}

      {/* Interactive Timeline Transport & Scrubber Deck */}
      {tab !== "sandbox" && (
        <div className="z-30 xl:sticky xl:top-2">
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
        </div>
      )}

      {tab === "floor" && (
        <div key="floor" className="anim-fade-up">
          <FactoryFloor
            scene={scene}
            paused={!isPlaying}
            selected={selectedMachine}
            onSelect={setSelectedMachine}
            title={`Live Factory Floor · ${parsed.meta.policy}`}
          />
        </div>
      )}

      {tab === "rl" && (
        <div key="rl" className="anim-fade-up flex flex-col gap-5">
          <MockNotice>
            No policy has been trained for this view. Decision times and actions come from the
            selected log; the action probabilities, network activations and reward breakdown are
            illustrative.
          </MockNotice>
          <RLBrain decisions={decisions} currentTime={currentTime} n={parsed.meta.n_machines} paused={!isPlaying} />
        </div>
      )}

      {tab === "arena" && (
        <div key="arena" className="anim-fade-up flex flex-col gap-5">
          <MockNotice>
            The training curves are synthetic, not from any PPO run. The policy race replays the{" "}
            {isMock ? "mock" : "real"} episodes for {parsed.meta.instance}.
          </MockNotice>
          <TrainingArena curves={curves} race={raceEntries} currentTime={currentTime} />
        </div>
      )}

      {tab === "sandbox" && (
        <div key="sandbox" className="anim-fade-up flex flex-col gap-5">
          <MockNotice>
            A toy simulator running in the browser. It is not the RAJS simulator and does not use
            the C-MAPSS health model.
          </MockNotice>
          <Sandbox />
        </div>
      )}

      {tab === "overview" && (
      <div key="overview" className="anim-fade-up flex flex-col gap-5">
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
                {isMock && (
                  <span className="ml-2">
                    <MockBadge />
                  </span>
                )}
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
                  <th className="pb-2.5 px-2">PMs</th>
                  <th className="pb-2.5 px-2">Breakdowns</th>
                  <th className="pb-2.5 pl-2">Downtime</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {comparisonRows.map((row) => {
                  const isCurrent = activeId === row.id;
                  return (
                    <tr
                      key={row.id}
                      onClick={() => selectEpisode(row.id)}
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
                          <span>{row.label}</span>
                          {row.meta.tau != null && (
                            <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
                              τ={row.meta.tau}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-2 font-bold text-cyan-300">
                        {row.makespan}u
                      </td>
                      <td className="py-3 px-2 text-emerald-300">
                        {row.maintenances}
                      </td>
                      <td className="py-3 px-2 text-rose-300">
                        {row.breakdowns}
                      </td>
                      <td className="py-3 pl-2 text-slate-300">
                        {formatDowntime(row)}
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
                  {ev.machine !== undefined && `M${ev.machine} `}
                  {ev.job !== undefined && `J${ev.job}·O${ev.op} `}
                  {ev.risk !== undefined && `r=${ev.risk.toFixed(2)}`}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      </div>
      )}
    </main>
  );
}
