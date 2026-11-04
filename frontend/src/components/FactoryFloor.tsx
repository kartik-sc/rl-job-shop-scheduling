"use client";

import React, { useMemo, useState } from "react";
import * as d3 from "d3";
import { Scene, SceneJob, SceneMachine } from "@/lib/scene";
import { clamp, easeInOut, jobColor, riskColor, shade } from "@/lib/viz";

const CELL_W = 270;
const CELL_H = 235;
const BODY_W = 150;
const BODY_H = 96;

export function layoutFor(n: number) {
  const cols = n <= 4 ? 2 : n <= 9 ? 3 : 4;
  const rows = Math.ceil(n / cols);
  const pos = (i: number) => ({
    x: 30 + (i % cols) * CELL_W + CELL_W / 2,
    y: 40 + Math.floor(i / cols) * CELL_H + 80,
  });
  return { cols, rows, pos, width: cols * CELL_W + 60, height: rows * CELL_H + 150 };
}

function gearPath(ro: number, ri: number, teeth: number) {
  const pts: string[] = [];
  const step = (Math.PI * 2) / teeth;
  for (let i = 0; i < teeth; i++) {
    const a = i * step;
    const p = [
      [ri, a - step * 0.5],
      [ro, a - step * 0.22],
      [ro, a + step * 0.22],
      [ri, a + step * 0.5],
    ];
    for (const [r, ang] of p)
      pts.push(`${(Math.cos(ang) * r).toFixed(2)},${(Math.sin(ang) * r).toFixed(2)}`);
  }
  return `M${pts.join("L")}Z`;
}
const GEAR = gearPath(21, 16, 10);

type Pt = [number, number];
function bezier(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: number): Pt {
  const u = 1 - t;
  return [
    u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
  ];
}

export function JobCube({
  color,
  label,
  size = 11,
  dim = false,
  glow = false,
}: {
  color: string;
  label?: string | number;
  size?: number;
  dim?: boolean;
  glow?: boolean;
}) {
  const s = size;
  const top = `0,${-s} ${s * 0.9},${-s * 0.5} 0,0 ${-s * 0.9},${-s * 0.5}`;
  const left = `${-s * 0.9},${-s * 0.5} 0,0 0,${s * 0.95} ${-s * 0.9},${s * 0.45}`;
  const right = `${s * 0.9},${-s * 0.5} 0,0 0,${s * 0.95} ${s * 0.9},${s * 0.45}`;
  return (
    <g opacity={dim ? 0.3 : 1} style={glow ? { filter: `drop-shadow(0 0 6px ${color})` } : undefined}>
      <polygon points={left} fill={shade(color, -0.9)} />
      <polygon points={right} fill={shade(color, -0.4)} />
      <polygon points={top} fill={shade(color, 0.4)} stroke={shade(color, 0.9)} strokeWidth={0.6} />
      {label !== undefined && (
        <text
          y={-s * 0.35}
          textAnchor="middle"
          fontSize={s * 0.62}
          fontWeight={700}
          fill="#06101c"
          style={{ fontFamily: "var(--font-mono)" }}
        >
          {label}
        </text>
      )}
    </g>
  );
}

function Machine({
  m,
  cx,
  cy,
  selected,
  job,
}: {
  m: SceneMachine;
  cx: number;
  cy: number;
  selected: boolean;
  job: SceneJob | undefined;
}) {
  const rc = riskColor(m.risk);
  const aura = d3.color(rc)?.copy({ opacity: 0.35 + m.risk * 0.4 }).formatRgb() ?? rc;
  const stateColor =
    m.state === "breakdown"
      ? "#fb7185"
      : m.state === "maintenance"
        ? "#34d399"
        : m.state === "operating"
          ? "#38bdf8"
          : "#64748b";
  const fillA =
    m.state === "breakdown" ? "#3b0f1c" : m.state === "maintenance" ? "#0b2f26" : m.state === "operating" ? "#0c2a44" : "#121b2d";
  const fillB =
    m.state === "breakdown" ? "#1b0710" : m.state === "maintenance" ? "#051a15" : m.state === "operating" ? "#07182a" : "#0a1120";
  const gid = `mg-${m.id}`;
  const label =
    m.state === "operating"
      ? "RUNNING"
      : m.state === "maintenance"
        ? "PM · SERVICING"
        : m.state === "breakdown"
          ? "FAILURE"
          : "STANDBY";
  const ringProps = {
    x: -BODY_W / 2,
    y: -BODY_H / 2,
    width: BODY_W,
    height: BODY_H,
    rx: 14,
    fill: "none",
    strokeWidth: 2,
  };

  return (
    <g transform={`translate(${cx},${cy})`} style={{ cursor: "pointer" }}>
      <g className={m.state === "breakdown" ? "fx-shake" : ""}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={fillA} />
            <stop offset="100%" stopColor={fillB} />
          </linearGradient>
        </defs>

        <ellipse cx={0} cy={BODY_H / 2 + 6} rx={BODY_W / 2 + 14} ry={11} fill="#000" opacity={0.45} />
        <rect x={-BODY_W / 2 - 8} y={BODY_H / 2 - 6} width={BODY_W + 16} height={12} rx={4} fill="#0d1526" stroke="#1e293b" />

        {m.state === "breakdown" && (
          <>
            <rect className="fx-ring" {...ringProps} stroke="#fb7185" />
            <rect className="fx-ring" style={{ animationDelay: "0.7s" }} {...ringProps} stroke="#fb7185" />
          </>
        )}
        {m.state === "maintenance" && (
          <>
            <rect className="fx-ring" {...ringProps} stroke="#34d399" />
            <rect className="fx-ring" style={{ animationDelay: "0.9s" }} {...ringProps} stroke="#34d399" />
          </>
        )}

        <g style={{ filter: `drop-shadow(0 0 ${5 + m.risk * 16}px ${aura})`, transition: "filter 0.5s" }}>
          <rect
            x={-BODY_W / 2}
            y={-BODY_H / 2}
            width={BODY_W}
            height={BODY_H}
            rx={14}
            fill={`url(#${gid})`}
            stroke={selected ? "#fff" : stateColor}
            strokeWidth={selected ? 2.2 : 1.4}
            strokeOpacity={selected ? 0.95 : 0.7}
          />
          <rect x={-BODY_W / 2 + 3} y={-BODY_H / 2 + 3} width={BODY_W - 6} height={22} rx={10} fill="rgba(255,255,255,0.04)" />
        </g>

        <text x={-BODY_W / 2 + 12} y={-BODY_H / 2 + 17} fontSize={12} fontWeight={700} fill="#e2e8f0" style={{ fontFamily: "var(--font-display)" }}>
          M{String(m.id).padStart(2, "0")}
        </text>
        <text x={-BODY_W / 2 + 46} y={-BODY_H / 2 + 17} fontSize={8.5} letterSpacing={1.2} fill={stateColor} style={{ fontFamily: "var(--font-mono)" }}>
          {label}
        </text>
        <circle cx={BODY_W / 2 - 14} cy={-BODY_H / 2 + 13} r={4} fill={stateColor} className={m.state === "idle" ? "" : "fx-blink"} />

        <g transform="translate(-34,-6)">
          {m.state === "breakdown" ? (
            <g>
              <path d={GEAR} fill="#3b1220" stroke="#fb7185" strokeWidth={1.2} transform="rotate(14)" opacity={0.7} />
              <path d="M3 -16 L-7 3 L0 3 L-3 16 L8 -4 L1 -4 Z" fill="#fde047" stroke="#f59e0b" strokeWidth={0.8} className="fx-flicker" />
            </g>
          ) : m.state === "maintenance" ? (
            <g>
              <path d={GEAR} fill="#06251d" stroke="#34d399" strokeWidth={1.2} className="fx-spin-rev" />
              <g className="fx-orbit">
                <g transform="translate(0,-26) rotate(40)">
                  <rect x={-2.2} y={-9} width={4.4} height={20} rx={2} fill="#a7f3d0" />
                  <circle cx={0} cy={-10} r={5.2} fill="#a7f3d0" />
                  <rect x={-1.6} y={-16} width={3.2} height={7} fill="#06251d" />
                </g>
              </g>
            </g>
          ) : (
            <g>
              <path
                d={GEAR}
                fill={m.state === "operating" ? "#0e3a5c" : "#17233a"}
                stroke={stateColor}
                strokeWidth={1.2}
                className={m.state === "operating" ? "fx-spin" : ""}
              />
              <circle r={6} fill="#050b16" stroke={stateColor} strokeWidth={1} />
            </g>
          )}
        </g>

        <g transform="translate(34,-2)">
          <rect x={-30} y={-6} width={60} height={26} rx={6} fill="#050b16" stroke="#1e293b" />
          {m.state === "operating" && job && (
            <g transform="translate(0,6)">
              <g className="fx-bob">
                <JobCube color={jobColor(job.id)} label={job.id} size={13} glow />
              </g>
            </g>
          )}
          {m.state === "operating" && (
            <g className="fx-belt" opacity={0.55}>
              {[0, 1, 2, 3, 4].map((i) => (
                <rect key={i} x={-28 + i * 14} y={16} width={7} height={2} fill={stateColor} />
              ))}
            </g>
          )}
          {m.state === "breakdown" && (
            <text y={11} textAnchor="middle" fontSize={9} fill="#fb7185" style={{ fontFamily: "var(--font-mono)" }}>
              OFFLINE
            </text>
          )}
        </g>

        <rect x={-BODY_W / 2 + 12} y={BODY_H / 2 - 28} width={BODY_W - 24} height={4} rx={2} fill="#0a1220" />
        <rect
          x={-BODY_W / 2 + 12}
          y={BODY_H / 2 - 28}
          width={(BODY_W - 24) * clamp(m.progress, 0, 1)}
          height={4}
          rx={2}
          fill={stateColor}
        />
        <rect x={-BODY_W / 2 + 12} y={BODY_H / 2 - 17} width={BODY_W - 56} height={6} rx={3} fill="#0a1220" stroke="#1e293b" strokeWidth={0.6} />
        <rect
          x={-BODY_W / 2 + 12}
          y={BODY_H / 2 - 17}
          width={(BODY_W - 56) * clamp(m.risk, 0, 1)}
          height={6}
          rx={3}
          fill={rc}
          style={{ transition: "width 0.4s, fill 0.4s" }}
        />
        <text x={BODY_W / 2 - 12} y={BODY_H / 2 - 10} textAnchor="end" fontSize={10} fontWeight={700} fill={rc} style={{ fontFamily: "var(--font-mono)" }}>
          {(m.risk * 100).toFixed(0)}%
        </text>

        {m.state === "breakdown" &&
          [0, 1, 2, 3].map((i) => (
            <circle key={i} className="fx-smoke" cx={-22 + i * 14} cy={-BODY_H / 2} r={6 + (i % 2) * 3} fill="#475569" style={{ animationDelay: `${i * 0.5}s` }} />
          ))}
        {m.state === "breakdown" &&
          [0, 1, 2, 3, 4, 5].map((i) => (
            <circle
              key={`s${i}`}
              className="fx-spark"
              cx={-34}
              cy={-6}
              r={1.8}
              fill="#fde047"
              style={{ animationDelay: `${i * 0.17}s`, ["--sx" as string]: `${(i - 2.5) * 11}px` }}
            />
          ))}
      </g>
    </g>
  );
}

export default function FactoryFloor({
  scene,
  paused,
  selected,
  onSelect,
  actions,
  title = "Live Factory Floor",
  hint,
}: {
  scene: Scene;
  paused: boolean;
  selected: number | null;
  onSelect: (m: number | null) => void;
  actions?: (m: SceneMachine) => React.ReactNode;
  title?: string;
  hint?: string;
}) {
  const n = scene.machines.length;
  const L = useMemo(() => layoutFor(n), [n]);
  const [hoverJob, setHoverJob] = useState<number | null>(null);

  const jobByMachine = new Map<number, SceneJob>();
  for (const j of scene.jobs) if (j.status === "processing" && j.machine !== null) jobByMachine.set(j.machine, j);
  const queueCounts = new Map<number, number>();
  for (const j of scene.jobs)
    if (j.status === "queued" && j.machine !== null) queueCounts.set(j.machine, (queueCounts.get(j.machine) ?? 0) + 1);

  const trayPos = (m: number, idx: number): Pt => {
    const p = L.pos(m);
    return [p.x - BODY_W / 2 + 12 + idx * 19, p.y + BODY_H / 2 + 32];
  };
  const outPort = (m: number): Pt => {
    const p = L.pos(m);
    return [p.x, p.y + BODY_H / 2 + 4];
  };
  const curve = (a: number, b: number, dstIdx: number) => {
    const p0 = outPort(a);
    const p3 = trayPos(b, dstIdx);
    const p1: Pt = [p0[0], p0[1] + 70];
    const p2: Pt = [p3[0], p3[1] + (p3[1] > p0[1] + 40 ? -70 : 70)];
    return { p0, p1, p2, p3 };
  };

  const dock = { x: L.width / 2, y: L.height - 55 };
  const sel = selected !== null ? scene.machines[selected] : null;
  const activeEdges = new Set(scene.jobs.filter((j) => j.status === "transit").map((j) => `${j.from}-${j.to}`));

  return (
    <div className={`rajs-panel relative overflow-hidden rounded-2xl ${paused ? "anim-paused" : ""}`}>
      <div className="pointer-events-none absolute inset-0 floor-grid opacity-70" />
      <div className="relative flex flex-wrap items-center justify-between gap-3 px-5 pt-4">
        <div>
          <h2 className="font-display text-base font-bold text-white">{title}</h2>
          <p className="font-mono text-[11px] text-slate-400">
            {hint ?? "Click a machine to inspect · cubes are jobs · glow = failure risk"}
          </p>
        </div>
        <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
          <Legend c="#38bdf8" l="running" />
          <Legend c="#34d399" l="PM" />
          <Legend c="#fb7185" l="failed" />
          <Legend c="#64748b" l="idle" />
        </div>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${L.width} ${L.height}`} className="block w-full" onClick={() => onSelect(null)}>
          {scene.flows.map(([a, b]) => {
            const c = curve(a, b, 0);
            const d = `M${c.p0} C${c.p1} ${c.p2} ${c.p3}`;
            const active = activeEdges.has(`${a}-${b}`);
            return (
              <g key={`${a}-${b}`}>
                <path d={d} fill="none" stroke="#0f1b30" strokeWidth={10} strokeLinecap="round" />
                <path
                  d={d}
                  fill="none"
                  stroke={active ? "#38bdf8" : "#25344f"}
                  strokeWidth={active ? 2.2 : 1.4}
                  strokeDasharray="5 7"
                  className="fx-march"
                  strokeLinecap="round"
                />
              </g>
            );
          })}

          {scene.machines.map((m) => {
            const [tx, ty] = trayPos(m.id, 0);
            return (
              <g key={`tray-${m.id}`}>
                <rect x={tx - 12} y={ty - 13} width={BODY_W - 2} height={26} rx={8} fill="#08101f" stroke="#1a2740" />
                <text x={tx - 6} y={ty + 24} fontSize={8} fill="#475569" style={{ fontFamily: "var(--font-mono)" }}>
                  INPUT BUFFER · {queueCounts.get(m.id) ?? 0}
                </text>
              </g>
            );
          })}

          {scene.machines.map((m) => {
            const p = L.pos(m.id);
            return (
              <g
                key={m.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(selected === m.id ? null : m.id);
                }}
              >
                <Machine m={m} cx={p.x} cy={p.y} selected={selected === m.id} job={jobByMachine.get(m.id)} />
              </g>
            );
          })}

          <g transform={`translate(${dock.x},${dock.y})`}>
            <rect x={-210} y={-22} width={420} height={44} rx={12} fill="#08101f" stroke="#1a2740" />
            <text x={-196} y={-8} fontSize={8.5} letterSpacing={1.5} fill="#475569" style={{ fontFamily: "var(--font-mono)" }}>
              SHIPPING DOCK · {scene.doneJobs}/{scene.jobs.length} JOBS COMPLETE
            </text>
          </g>

          {scene.jobs.map((j) => {
            const color = jobColor(j.id);
            const dim = hoverJob !== null && hoverJob !== j.id;
            if (j.status === "queued" && j.machine !== null) {
              const [x, y] = trayPos(j.machine, Math.min(j.queueIndex, 7));
              return (
                <g key={j.id} className="job-slide" style={{ transform: `translate(${x}px,${y + 3}px)` }}>
                  <JobCube color={color} label={j.id} size={9} dim={dim} glow={hoverJob === j.id} />
                </g>
              );
            }
            if (j.status === "blocked" && j.machine !== null) {
              const p = L.pos(j.machine);
              return (
                <g key={j.id} style={{ transform: `translate(${p.x + 34}px,${p.y + 6}px)` }}>
                  <g className="fx-flicker">
                    <JobCube color="#fb7185" label="!" size={13} dim={dim} />
                  </g>
                </g>
              );
            }
            if (j.status === "transit" && j.from !== undefined && j.to !== undefined) {
              const dstIdx = queueCounts.get(j.to) ?? 0;
              const c = curve(j.from, j.to, Math.min(dstIdx, 7));
              const [x, y] = bezier(c.p0, c.p1, c.p2, c.p3, easeInOut(clamp(j.p ?? 0, 0, 1)));
              return (
                <g key={j.id} style={{ transform: `translate(${x}px,${y + 3}px)` }}>
                  <g className="fx-bob">
                    <JobCube color={color} label={j.id} size={10} dim={dim} glow />
                  </g>
                </g>
              );
            }
            if (j.status === "done") {
              const idx = scene.jobs.filter((q) => q.status === "done" && q.id <= j.id).length - 1;
              return (
                <g key={j.id} className="job-slide" style={{ transform: `translate(${dock.x - 170 + idx * 26}px,${dock.y + 8}px)` }}>
                  <JobCube color={color} label="✓" size={10} dim={dim} />
                </g>
              );
            }
            return null;
          })}
        </svg>

        {sel && (
          <div className="anim-pop absolute right-4 top-4 w-64 rounded-xl border border-white/10 bg-slate-950/90 p-4 shadow-2xl backdrop-blur-md">
            <div className="flex items-center justify-between">
              <span className="font-display text-sm font-bold text-white">Machine {sel.id}</span>
              <button className="text-xs text-slate-500 hover:text-white" onClick={() => onSelect(null)}>
                ✕
              </button>
            </div>
            <div className="mt-1 font-mono text-[11px] text-slate-400">{sel.detail}</div>
            <div className="mt-3 flex items-center gap-3">
              <RiskDial risk={sel.risk} />
              <div className="font-mono text-[11px] leading-5 text-slate-300">
                <div>
                  state: <b className="text-white">{sel.state}</b>
                </div>
                <div>
                  progress: <b className="text-white">{(sel.progress * 100).toFixed(0)}%</b>
                </div>
                <div>
                  job: <b className="text-white">{sel.jobId ?? "—"}</b>
                </div>
              </div>
            </div>
            {actions && <div className="mt-3 flex flex-wrap gap-2">{actions(sel)}</div>}
          </div>
        )}
      </div>

      <div className="relative flex flex-wrap gap-2 px-5 pb-4">
        {scene.jobs.map((j) => (
          <div
            key={j.id}
            onMouseEnter={() => setHoverJob(j.id)}
            onMouseLeave={() => setHoverJob(null)}
            className="flex cursor-default items-center gap-2 rounded-lg border border-white/10 bg-slate-950/60 px-2.5 py-1.5 transition hover:border-white/30"
          >
            <svg width="22" height="22" viewBox="-14 -14 28 28">
              <g transform="translate(0,6)">
                <JobCube color={jobColor(j.id)} label={j.id} size={11} />
              </g>
            </svg>
            <div className="flex gap-1">
              {Array.from({ length: j.nOps }).map((_, i) => (
                <span
                  key={i}
                  className="h-1.5 w-4 rounded-full transition-colors"
                  style={{
                    background:
                      i < j.opIdx ? jobColor(j.id) : i === j.opIdx && j.status === "processing" ? "#ffffff" : "#1e293b",
                  }}
                />
              ))}
            </div>
            <span className="font-mono text-[10px] uppercase text-slate-500">{j.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Legend({ c, l }: { c: string; l: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ background: c, boxShadow: `0 0 8px ${c}` }} />
      {l}
    </span>
  );
}

export function RiskDial({ risk, size = 64 }: { risk: number; size?: number }) {
  const r = size / 2 - 6;
  const circ = Math.PI * 2 * r * 0.75;
  const col = riskColor(risk);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <g transform={`rotate(135 ${size / 2} ${size / 2})`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#17233a" strokeWidth={6} strokeDasharray={`${circ} 999`} strokeLinecap="round" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={col}
          strokeWidth={6}
          strokeDasharray={`${circ * clamp(risk, 0, 1)} 999`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray .3s, stroke .3s", filter: `drop-shadow(0 0 4px ${col})` }}
        />
      </g>
      <text x={size / 2} y={size / 2 + 4} textAnchor="middle" fontSize={size * 0.22} fontWeight={700} fill={col} style={{ fontFamily: "var(--font-mono)" }}>
        {(risk * 100).toFixed(0)}
      </text>
    </svg>
  );
}
