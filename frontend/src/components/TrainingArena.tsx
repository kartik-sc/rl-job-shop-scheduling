"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { Flag, Play, RotateCcw, Trophy } from "lucide-react";
import { TrainingCurves } from "@/lib/rl";
import { ParsedEpisode } from "@/types/eventLog";
import { buildEpisodeScene, buildFlows, buildRoutes } from "@/lib/scene";

const COLORS = { full: "#f0abfc", blind: "#fb923c", threshold: "#38bdf8" };

export interface RaceEntry {
  key: string;
  name: string;
  color: string;
  parsed: ParsedEpisode;
}

export default function TrainingArena({
  curves,
  race,
  currentTime,
}: {
  curves: TrainingCurves;
  race: RaceEntry[];
  currentTime: number;
}) {
  const total = curves.iters.length;
  const [iter, setIter] = useState(total - 1);
  const [running, setRunning] = useState(false);
  const raf = useRef(0);

  useEffect(() => {
    if (!running) return;
    let last = performance.now();
    let acc = iter >= total - 1 ? 0 : iter;
    const loop = (now: number) => {
      acc += ((now - last) / 1000) * 38;
      last = now;
      if (acc >= total - 1) {
        setIter(total - 1);
        setRunning(false);
        return;
      }
      setIter(Math.floor(acc));
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, total]);

  const lanes = useMemo(
    () =>
      race.map((r) => {
        const routes = buildRoutes(r.parsed);
        return { ...r, routes, flows: buildFlows(routes) };
      }),
    [race]
  );

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      <div className="rajs-panel rounded-2xl p-5 xl:col-span-8">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-base font-bold text-white">Training arena</h2>
            <p className="font-mono text-[11px] text-slate-400">Mean episode return per PPO update · synthetic curves until real runs are logged</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setIter(0);
                setRunning(true);
              }}
              className="flex h-9 items-center gap-2 rounded-xl bg-gradient-to-r from-fuchsia-500 to-indigo-500 px-4 text-sm font-semibold text-white shadow-lg shadow-fuchsia-500/20 active:scale-95"
            >
              <Play className="h-4 w-4 fill-current" /> Train
            </button>
            <button
              onClick={() => {
                setRunning(false);
                setIter(total - 1);
              }}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-700 bg-slate-900 text-slate-300 hover:text-white"
              title="Skip to end"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>

        <CurveChart title="Episode return ↑" iters={curves.iters} iter={iter} series={[
          { name: "Risk-aware PPO", color: COLORS.full, v: curves.full },
          { name: "Blind PPO", color: COLORS.blind, v: curves.blind },
          { name: "Threshold + SPT", color: COLORS.threshold, v: curves.threshold, dashed: true },
        ]} />
        <div className="mt-4">
          <CurveChart title="Breakdowns per episode ↓" iters={curves.iters} iter={iter} height={150} series={[
            { name: "Risk-aware PPO", color: COLORS.full, v: curves.fullBreak },
            { name: "Blind PPO", color: COLORS.blind, v: curves.blindBreak },
            { name: "Threshold + SPT", color: COLORS.threshold, v: curves.thresholdBreak, dashed: true },
          ]} />
        </div>

        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-slate-900">
          <div className="h-full rounded-full bg-gradient-to-r from-fuchsia-400 to-indigo-400" style={{ width: `${(iter / (total - 1)) * 100}%` }} />
        </div>
        <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
          <span>update {iter}</span>
          <span>{total - 1}</span>
        </div>
      </div>

      <div className="flex flex-col gap-5 xl:col-span-4">
        <div className="rajs-panel rounded-2xl p-5">
          <div className="flex items-center gap-2">
            <Flag className="h-4 w-4 text-amber-300" />
            <h3 className="font-display text-sm font-bold text-white">Policy race · same instance</h3>
          </div>
          <p className="mb-3 mt-1 font-mono text-[11px] text-slate-400">Driven by the global transport · t = {currentTime.toFixed(0)}u</p>
          <div className="space-y-4">
            {lanes.map((l) => {
              const sc = buildEpisodeScene(l.parsed, l.routes, l.flows, currentTime);
              const finished = currentTime >= l.parsed.makespan;
              return (
                <div key={l.key}>
                  <div className="flex items-center justify-between font-mono text-[11px]">
                    <span style={{ color: l.color }} className="font-semibold">{l.name}</span>
                    <span className="text-slate-400">{finished ? <span className="flex items-center gap-1 text-emerald-300"><Trophy className="h-3 w-3" />{l.parsed.makespan}u</span> : `${sc.completedOps}/${sc.totalOps} ops`}</span>
                  </div>
                  <div className="relative mt-1.5 h-3 overflow-hidden rounded-full bg-slate-900">
                    <div className="h-full rounded-full shimmer" style={{ width: `${(sc.completedOps / sc.totalOps) * 100}%`, background: `linear-gradient(90deg, ${l.color}55, ${l.color})`, transition: "width .15s linear" }} />
                  </div>
                  <div className="mt-2 flex gap-1.5">
                    {sc.machines.map((m) => (
                      <div
                        key={m.id}
                        title={`M${m.id} ${m.state}`}
                        className={`h-5 flex-1 rounded-md border text-center font-mono text-[9px] leading-5 ${
                          m.state === "operating"
                            ? "border-sky-400/50 bg-sky-500/20 text-sky-200"
                            : m.state === "maintenance"
                              ? "border-emerald-400/60 bg-emerald-500/25 text-emerald-200"
                              : m.state === "breakdown"
                                ? "fx-flicker border-rose-400/60 bg-rose-500/30 text-rose-200"
                                : "border-slate-800 bg-slate-900 text-slate-600"
                        }`}
                      >
                        {m.id}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rajs-panel rounded-2xl p-5">
          <h3 className="font-display text-sm font-bold text-white">Live readout @ update {iter}</h3>
          <div className="mt-3 space-y-2 font-mono text-xs">
            {[
              { n: "Risk-aware PPO", c: COLORS.full, r: curves.full[iter], b: curves.fullBreak[iter] },
              { n: "Blind PPO", c: COLORS.blind, r: curves.blind[iter], b: curves.blindBreak[iter] },
              { n: "Threshold + SPT", c: COLORS.threshold, r: curves.threshold[iter], b: curves.thresholdBreak[iter] },
            ].map((x) => (
              <div key={x.n} className="rajs-card-inner flex items-center justify-between rounded-lg px-3 py-2">
                <span className="flex items-center gap-2 text-slate-200"><span className="h-2 w-2 rounded-full" style={{ background: x.c, boxShadow: `0 0 8px ${x.c}` }} />{x.n}</span>
                <span className="text-slate-300">G={x.r.toFixed(1)} · bd={x.b.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function CurveChart({
  title,
  iters,
  iter,
  series,
  height = 230,
}: {
  title: string;
  iters: number[];
  iter: number;
  series: { name: string; color: string; v: number[]; dashed?: boolean }[];
  height?: number;
}) {
  const W = 760;
  const pad = { l: 40, r: 14, t: 14, b: 22 };
  const all = series.flatMap((s) => s.v);
  const x = d3.scaleLinear().domain([0, iters.length - 1]).range([pad.l, W - pad.r]);
  const y = d3.scaleLinear().domain([Math.min(...all), Math.max(...all)]).nice().range([height - pad.b, pad.t]);
  const smooth = (v: number[]) => v.map((_, i) => d3.mean(v.slice(Math.max(0, i - 4), i + 1)) ?? 0);
  const line = d3.line<number>().x((_, i) => x(i)).y((v) => y(v)).curve(d3.curveCatmullRom);

  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2 px-1">
        <span className="font-mono text-[11px] uppercase tracking-wider text-slate-400">{title}</span>
        <div className="flex gap-3 font-mono text-[10px] text-slate-400">
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5"><span className="h-0.5 w-3" style={{ background: s.color }} />{s.name}</span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${height}`} className="block w-full">
        {y.ticks(4).map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#17233a" />
            <text x={pad.l - 6} y={y(t) + 3} textAnchor="end" fontSize={9} fill="#64748b" style={{ fontFamily: "var(--font-mono)" }}>{t}</text>
          </g>
        ))}
        {series.map((s) => {
          const upto = s.v.slice(0, iter + 1);
          const sm = smooth(upto);
          return (
            <g key={s.name}>
              <path d={line(upto) ?? ""} fill="none" stroke={s.color} strokeOpacity={0.22} strokeWidth={1} />
              <path d={line(sm) ?? ""} fill="none" stroke={s.color} strokeWidth={2.3} strokeDasharray={s.dashed ? "6 5" : undefined} style={{ filter: `drop-shadow(0 0 5px ${s.color}88)` }} />
              <circle cx={x(iter)} cy={y(sm[sm.length - 1] ?? 0)} r={4} fill={s.color} style={{ filter: `drop-shadow(0 0 6px ${s.color})` }} />
            </g>
          );
        })}
        <line x1={x(iter)} x2={x(iter)} y1={pad.t} y2={height - pad.b} stroke="#fff" strokeOpacity={0.12} />
      </svg>
    </div>
  );
}
