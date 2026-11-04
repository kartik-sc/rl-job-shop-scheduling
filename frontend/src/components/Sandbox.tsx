"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Bolt, Pause, Play, Plus, RotateCcw, Wrench } from "lucide-react";
import FactoryFloor from "@/components/FactoryFloor";
import {
  Sim,
  SimPolicy,
  addRushJob,
  forceBreakdown,
  forceMaintenance,
  newSim,
  simScene,
  step,
} from "@/lib/sandboxSim";

const POLICIES: { key: SimPolicy; name: string; desc: string }[] = [
  { key: "manual", name: "Manual", desc: "You are the scheduler: service machines yourself." },
  { key: "threshold", name: "Threshold + SPT", desc: "Auto PM at risk ≥ 70%, dispatch shortest job first." },
  { key: "riskaware", name: "Risk-aware agent", desc: "Heuristic stand-in for the PPO agent: servicing early when queues are short." },
];

export default function Sandbox() {
  const simRef = useRef<Sim>(newSim());
  const [, force] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [policy, setPolicy] = useState<SimPolicy>("manual");
  const [stress, setStress] = useState(1.4);
  const [selected, setSelected] = useState<number | null>(null);
  const policyRef = useRef(policy);
  const stressRef = useRef(stress);
  const speedRef = useRef(speed);
  policyRef.current = policy;
  stressRef.current = stress;
  speedRef.current = speed;

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const s = simRef.current;
      if (s.finishedAt === null) {
        const sub = 4;
        for (let i = 0; i < sub; i++) step(s, (dt * 5 * speedRef.current) / sub, policyRef.current, stressRef.current);
      }
      force((x) => x + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const s = simRef.current;
  const scene = simScene(s);
  const done = s.finishedAt !== null;
  const maxRisk = useMemo(() => Math.max(...s.machines.map((m) => m.risk)), [s.t]); // eslint-disable-line react-hooks/exhaustive-deps

  const reset = () => {
    simRef.current = newSim();
    setSelected(null);
    force((x) => x + 1);
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="rajs-panel flex flex-wrap items-center justify-between gap-4 rounded-2xl p-4">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setPlaying((p) => !p)}
            className={`flex h-10 items-center gap-2 rounded-xl px-4 text-sm font-semibold text-white transition active:scale-95 ${
              playing ? "bg-gradient-to-r from-amber-500 to-orange-500" : "bg-gradient-to-r from-cyan-500 to-blue-600"
            }`}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {playing ? "Pause" : "Run"}
          </button>
          <button onClick={reset} className="flex h-10 items-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-3 text-sm text-slate-200 hover:border-slate-500">
            <RotateCcw className="h-4 w-4" /> Reset
          </button>
          <button
            onClick={() => {
              addRushJob(simRef.current);
              force((x) => x + 1);
            }}
            className="flex h-10 items-center gap-2 rounded-xl border border-violet-500/40 bg-violet-500/10 px-3 text-sm text-violet-200 hover:bg-violet-500/20"
          >
            <Plus className="h-4 w-4" /> Rush job
          </button>
          <div className="flex rounded-xl border border-slate-800 bg-slate-950/90 p-1 text-xs">
            {[1, 2, 4].map((v) => (
              <button
                key={v}
                onClick={() => setSpeed(v)}
                className={`rounded-lg px-2.5 py-1 font-mono font-semibold ${speed === v ? "bg-cyan-500/20 text-cyan-300" : "text-slate-400 hover:text-white"}`}
              >
                {v}×
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex rounded-xl border border-slate-800 bg-slate-950/90 p-1">
            {POLICIES.map((p) => (
              <button
                key={p.key}
                onClick={() => setPolicy(p.key)}
                title={p.desc}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                  policy === p.key ? "border border-cyan-400/40 bg-cyan-500/20 text-cyan-200" : "border border-transparent text-slate-400 hover:text-white"
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
            wear
            <input type="range" min={0.5} max={3} step={0.1} value={stress} onChange={(e) => setStress(Number(e.target.value))} className="w-28 accent-rose-400" />
            <span className="w-8 text-rose-300">{stress.toFixed(1)}×</span>
          </label>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat label="clock" value={`${s.t.toFixed(0)}u`} accent="text-cyan-300" />
        <Stat label="ops done" value={`${scene.completedOps}/${scene.totalOps}`} accent="text-indigo-300" />
        <Stat label="PMs" value={String(s.pms)} accent="text-emerald-300" />
        <Stat label="breakdowns" value={String(s.breakdowns)} accent="text-rose-300" />
        <Stat label="return" value={s.reward.toFixed(1)} accent={s.reward >= 0 ? "text-emerald-300" : "text-amber-300"} sub={`peak risk ${(maxRisk * 100).toFixed(0)}%`} />
      </div>

      <div className="relative">
        <FactoryFloor
          scene={scene}
          paused={!playing}
          selected={selected}
          onSelect={setSelected}
          title="Interactive Sandbox · you vs. the failure curve"
          hint="Click a machine → service it or sabotage it. Switch the policy and watch it react."
          actions={(m) => (
            <>
              <button
                disabled={m.state !== "idle"}
                onClick={() => {
                  forceMaintenance(simRef.current, m.id);
                  force((x) => x + 1);
                }}
                className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1.5 text-xs font-semibold text-emerald-200 hover:bg-emerald-500/20 disabled:opacity-30"
              >
                <Wrench className="h-3.5 w-3.5" /> Service
              </button>
              <button
                disabled={m.state === "breakdown"}
                onClick={() => {
                  forceBreakdown(simRef.current, m.id);
                  force((x) => x + 1);
                }}
                className="flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-2.5 py-1.5 text-xs font-semibold text-rose-200 hover:bg-rose-500/20 disabled:opacity-30"
              >
                <Bolt className="h-3.5 w-3.5" /> Break it
              </button>
            </>
          )}
        />
        {done && (
          <div className="anim-pop absolute inset-x-0 top-1/3 mx-auto w-fit rounded-2xl border border-emerald-400/40 bg-slate-950/90 px-8 py-5 text-center shadow-2xl backdrop-blur">
            <div className="font-display text-xl font-bold text-emerald-300">Episode complete</div>
            <div className="mt-1 font-mono text-sm text-slate-300">
              makespan {s.finishedAt?.toFixed(0)}u · {s.breakdowns} breakdowns · return {s.reward.toFixed(1)}
            </div>
            <button onClick={reset} className="mt-3 rounded-lg bg-emerald-500/20 px-4 py-1.5 text-sm font-semibold text-emerald-200 hover:bg-emerald-500/30">
              Try another policy
            </button>
          </div>
        )}
      </div>

      <div className="rajs-panel rounded-2xl p-4">
        <h3 className="mb-2 font-display text-sm font-bold text-white">Event feed</h3>
        <div className="grid max-h-40 grid-cols-1 gap-1 overflow-y-auto font-mono text-[11px] text-slate-300 sm:grid-cols-2">
          {s.log.slice(0, 12).map((l, i) => (
            <div key={`${l}-${i}`} className={`rajs-card-inner rounded-md px-2.5 py-1 ${l.includes("FAILED") ? "text-rose-300" : l.includes("maintenance") || l.includes("online") ? "text-emerald-300" : ""}`}>
              {l}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, accent, sub }: { label: string; value: string; accent: string; sub?: string }) {
  return (
    <div className="rajs-panel rounded-xl px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-xl font-bold ${accent}`}>{value}</div>
      {sub && <div className="font-mono text-[10px] text-slate-500">{sub}</div>}
    </div>
  );
}
