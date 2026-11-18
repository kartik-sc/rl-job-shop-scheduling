"use client";

import React, { useMemo } from "react";
import * as d3 from "d3";
import { Brain, Sparkles } from "lucide-react";
import { Decision, H1, H2, actionLabels, inputLabels, makeWeights } from "@/lib/rl";
import { clamp, riskColor } from "@/lib/viz";

const act = d3.scaleLinear<string>().domain([-1, 0, 1]).range(["#fb7185", "#0b1428", "#38bdf8"]).clamp(true);

export default function RLBrain({
  decisions,
  currentTime,
  n,
  paused,
}: {
  decisions: Decision[];
  currentTime: number;
  n: number;
  paused: boolean;
}) {
  const labels = useMemo(() => actionLabels(n), [n]);
  const inLabels = useMemo(() => inputLabels(n), [n]);
  const weights = useMemo(() => makeWeights(3 * n), [n]);

  let activeIdx = -1;
  for (let i = 0; i < decisions.length; i++) if (decisions[i].t <= currentTime) activeIdx = i;
  const d = activeIdx >= 0 ? decisions[activeIdx] : null;
  const nIn = 3 * n;
  const nOut = 2 * n + 1;

  // ---- network geometry
  const W = 860;
  const H = 360;
  const xs = [90, 330, 520, 700];
  const layerY = (count: number, i: number) => 28 + (i + 0.5) * ((H - 56) / count);
  const input = d ? d.state : new Array(nIn).fill(0);
  const h1 = d ? d.h1 : new Array(H1).fill(0);
  const h2 = d ? d.h2 : new Array(H2).fill(0);
  const probs = d ? d.probs : new Array(nOut).fill(1 / nOut);

  const edges = useMemo(() => {
    const e: { x1: number; y1: number; x2: number; y2: number; w: number; layer: number; i: number; j: number }[] = [];
    for (let i = 0; i < nIn; i++)
      for (let j = 0; j < H1; j++)
        e.push({ x1: xs[0], y1: layerY(nIn, i), x2: xs[1], y2: layerY(H1, j), w: weights.w1[i][j], layer: 0, i, j });
    for (let i = 0; i < H1; i++)
      for (let j = 0; j < H2; j++)
        e.push({ x1: xs[1], y1: layerY(H1, i), x2: xs[2], y2: layerY(H2, j), w: weights.w2[i][j], layer: 1, i, j });
    for (let i = 0; i < H2; i++)
      for (let j = 0; j < nOut; j++)
        e.push({
          x1: xs[2],
          y1: layerY(H2, i),
          x2: xs[3],
          y2: layerY(nOut, j),
          w: (((i * 31 + j * 17) % 13) / 6.5 - 1) * 0.9,
          layer: 2,
          i,
          j,
        });
    return e;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nIn, nOut, weights]);

  const srcAct = (layer: number, i: number) => (layer === 0 ? input[i] : layer === 1 ? h1[i] : h2[i]);

  const ranked = probs
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 6);

  const returns = decisions.map((x) => x.ret);
  const retMin = Math.min(0, ...returns);
  const retMax = Math.max(1, ...returns);

  return (
    <div className={`grid grid-cols-1 gap-5 xl:grid-cols-12 ${paused ? "anim-paused" : ""}`}>
      {/* LOOP + NETWORK */}
      <div className="flex flex-col gap-5 xl:col-span-8">
        <LoopDiagram d={d} />

        <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
          <div className="mb-1 flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4 text-fuchsia-300" />
              <h2 className="font-display text-sm font-bold text-white">Policy network · forward pass at decision {d ? d.idx + 1 : "—"}</h2>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">illustrative weights</span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="block w-full">
            {edges.map((e, k) => {
              const a = srcAct(e.layer, e.i);
              const s = e.w * a;
              const mag = Math.abs(s);
              const op = 0.03 + clamp(mag, 0, 1) * 0.55;
              const fire = d && mag > 0.35;
              return (
                <line
                  key={k}
                  x1={e.x1}
                  y1={e.y1}
                  x2={e.x2}
                  y2={e.y2}
                  stroke={s >= 0 ? "#38bdf8" : "#fb7185"}
                  strokeOpacity={op}
                  strokeWidth={0.4 + mag * 1.2}
                  className={fire ? "net-edge" : ""}
                  style={{ transition: "stroke-opacity .4s" }}
                />
              );
            })}

            {/* inputs */}
            {input.map((v: number, i: number) => {
              const grp = Math.floor(i / n);
              const col = grp === 0 ? riskColor(v) : grp === 1 ? (v > 0.5 ? "#38bdf8" : "#17233a") : act(v);
              return (
                <g key={i}>
                  <circle cx={xs[0]} cy={layerY(nIn, i)} r={5.2} fill={col} stroke="#1e293b" style={{ transition: "fill .4s" }} />
                  <text x={xs[0] - 11} y={layerY(nIn, i) + 3} textAnchor="end" fontSize={8} fill="#64748b" style={{ fontFamily: "var(--font-mono)" }}>
                    {inLabels[i]}
                  </text>
                </g>
              );
            })}
            {h1.map((v: number, i: number) => (
              <circle key={`a${i}`} cx={xs[1]} cy={layerY(H1, i)} r={8} fill={act(v)} stroke="#334155" style={{ transition: "fill .4s" }} />
            ))}
            {h2.map((v: number, i: number) => (
              <circle key={`b${i}`} cx={xs[2]} cy={layerY(H2, i)} r={9.5} fill={act(v)} stroke="#334155" style={{ transition: "fill .4s" }} />
            ))}
            {probs.map((p: number, i: number) => {
              const chosen = d?.chosen === i;
              const y = layerY(nOut, i);
              return (
                <g key={`o${i}`}>
                  {chosen && <circle cx={xs[3]} cy={y} r={15} fill="none" stroke="#f0abfc" strokeWidth={1.5} className="fx-ring" style={{ transformOrigin: `${xs[3]}px ${y}px`, transformBox: "view-box" }} />}
                  <circle
                    cx={xs[3]}
                    cy={y}
                    r={4 + p * 16}
                    fill={chosen ? "#f0abfc" : "#6366f1"}
                    fillOpacity={chosen ? 1 : 0.35 + p}
                    stroke={chosen ? "#fff" : "#334155"}
                    style={{ transition: "r .4s, fill .3s", filter: chosen ? "drop-shadow(0 0 8px #f0abfc)" : undefined }}
                  />
                  <text x={xs[3] + 24} y={y + 3} fontSize={9} fill={chosen ? "#f5d0fe" : "#64748b"} fontWeight={chosen ? 700 : 400} style={{ fontFamily: "var(--font-mono)" }}>
                    {labels[i]} · {(p * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}
            {[
              ["state sₜ", xs[0]],
              ["hidden 1 · tanh", xs[1]],
              ["hidden 2 · tanh", xs[2]],
              ["policy π(a|s)", xs[3]],
            ].map(([t, x]) => (
              <text key={t as string} x={x as number} y={14} textAnchor="middle" fontSize={9} letterSpacing={1.2} fill="#94a3b8" style={{ fontFamily: "var(--font-mono)" }}>
                {(t as string).toUpperCase()}
              </text>
            ))}
          </svg>
        </div>
      </div>

      {/* SIDE */}
      <div className="flex flex-col gap-5 xl:col-span-4">
        <div className="rajs-panel rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <h3 className="font-display text-sm font-bold text-white">Chosen action</h3>
            <span className="font-mono text-[11px] text-slate-500">
              step {d ? d.idx + 1 : 0}/{decisions.length}
            </span>
          </div>
          <div key={d?.idx ?? -1} className="anim-pop mt-3 rounded-xl border border-fuchsia-400/30 bg-fuchsia-500/10 p-3">
            <div className="font-display text-lg font-bold text-fuchsia-200">{d ? labels[d.chosen] : "Awaiting first decision"}</div>
            <div className="font-mono text-[11px] text-slate-400">
              {d ? `t=${d.t}u${d.job !== undefined ? ` · job ${d.job} op ${d.op}` : " · preventive maintenance"}` : "press play"}
            </div>
          </div>
          <div className="mt-4 space-y-1.5">
            {ranked.map(({ p, i }) => (
              <div key={i} className="flex items-center gap-2 font-mono text-[10px]">
                <span className={`w-24 shrink-0 truncate ${d?.chosen === i ? "text-fuchsia-200" : "text-slate-400"}`}>{labels[i]}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-900">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${p * 100}%`,
                      background: d?.chosen === i ? "linear-gradient(90deg,#c026d3,#f0abfc)" : "linear-gradient(90deg,#4f46e5,#818cf8)",
                      transition: "width .45s cubic-bezier(.22,1,.36,1)",
                    }}
                  />
                </div>
                <span className="w-9 text-right text-slate-300">{(p * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="rajs-panel rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <h3 className="font-display text-sm font-bold text-white">Reward signal</h3>
            <Sparkles className="h-4 w-4 text-amber-300" />
          </div>
          <div className="mt-2 flex items-end gap-4">
            <div key={`r${d?.idx}`} className="anim-pop">
              <div className="font-mono text-[10px] uppercase text-slate-500">rₜ</div>
              <div className={`font-mono text-3xl font-bold ${d && d.reward < -1 ? "text-rose-300" : d && d.reward > 0 ? "text-emerald-300" : "text-slate-200"}`}>
                {d ? (d.reward >= 0 ? "+" : "") + d.reward.toFixed(2) : "0.00"}
              </div>
            </div>
            <div>
              <div className="font-mono text-[10px] uppercase text-slate-500">return Gₜ</div>
              <div className="font-mono text-xl font-bold text-amber-200">{d ? d.ret.toFixed(2) : "0.00"}</div>
            </div>
          </div>
          <svg viewBox="0 0 300 90" className="mt-3 block w-full">
            <defs>
              <linearGradient id="retfill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#fbbf24" stopOpacity="0" />
              </linearGradient>
            </defs>
            {decisions.length > 1 && (() => {
              const x = d3.scaleLinear().domain([0, decisions.length - 1]).range([6, 294]);
              const y = d3.scaleLinear().domain([retMin, retMax]).range([82, 8]);
              const line = d3.line<number>().x((_, i) => x(i)).y((v) => y(v)).curve(d3.curveMonotoneX);
              const area = d3.area<number>().x((_, i) => x(i)).y0(82).y1((v) => y(v)).curve(d3.curveMonotoneX);
              const upto = returns.slice(0, Math.max(0, activeIdx + 1));
              return (
                <g>
                  <line x1={6} x2={294} y1={y(0)} y2={y(0)} stroke="#1e293b" strokeDasharray="3 4" />
                  <path d={line(returns) ?? ""} fill="none" stroke="#334155" strokeWidth={1.2} />
                  {upto.length > 1 && <path d={area(upto) ?? ""} fill="url(#retfill)" />}
                  {upto.length > 1 && <path d={line(upto) ?? ""} fill="none" stroke="#fbbf24" strokeWidth={2} />}
                  {d && <circle cx={x(activeIdx)} cy={y(d.ret)} r={4} fill="#fde68a" style={{ filter: "drop-shadow(0 0 6px #fbbf24)" }} />}
                </g>
              );
            })()}
          </svg>
          {d && (
            <div className="mt-2 grid grid-cols-2 gap-1.5 font-mono text-[10px]">
              <Chip label="time cost" v={d.parts.time} />
              <Chip label="op complete" v={d.parts.completions} />
              <Chip label="breakdown" v={d.parts.breakdown} />
              <Chip label="PM cost" v={d.parts.maint} />
            </div>
          )}
        </div>

        <div className="rajs-panel rounded-2xl p-5">
          <h3 className="font-display text-sm font-bold text-white">Decision trace</h3>
          <div className="mt-3 max-h-56 space-y-1 overflow-y-auto pr-1">
            {decisions
              .slice(0, activeIdx + 1)
              .slice(-12)
              .reverse()
              .map((x) => (
                <div key={x.idx} className="rajs-card-inner flex items-center justify-between rounded-lg px-2.5 py-1.5 font-mono text-[11px]">
                  <span className="text-cyan-300">t={x.t}</span>
                  <span className={x.kind === "maintain" ? "text-emerald-300" : "text-slate-200"}>{labels[x.chosen]}</span>
                  <span className={x.reward < 0 ? "text-rose-300" : "text-emerald-300"}>{x.reward.toFixed(1)}</span>
                </div>
              ))}
            {activeIdx < 0 && <div className="font-mono text-xs text-slate-500">No actions taken yet.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function Chip({ label, v }: { label: string; v: number }) {
  return (
    <div className="flex items-center justify-between rounded-md bg-slate-900/80 px-2 py-1">
      <span className="text-slate-500">{label}</span>
      <span className={v < 0 ? "text-rose-300" : v > 0 ? "text-emerald-300" : "text-slate-400"}>{v.toFixed(2)}</span>
    </div>
  );
}

function LoopDiagram({ d }: { d: Decision | null }) {
  const key = d?.idx ?? -1;
  const aPath = "M 250 70 C 330 20, 490 20, 570 70";
  const sPath = "M 570 150 C 490 205, 330 205, 250 150";
  return (
    <div className="rajs-panel relative overflow-hidden rounded-2xl p-4">
      <div className="mb-1 flex items-center justify-between px-1">
        <h2 className="font-display text-sm font-bold text-white">Agent ⇄ Environment loop</h2>
        <span className="font-mono text-[10px] uppercase tracking-wider text-slate-500">MDP · each decision fires a packet</span>
      </div>
      <svg viewBox="0 0 820 230" className="block w-full">
        <defs>
          <path id="aPath" d={aPath} />
          <path id="sPath" d={sPath} />
          <linearGradient id="agentGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#4c1d95" />
            <stop offset="100%" stopColor="#1e1b4b" />
          </linearGradient>
          <linearGradient id="envGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#0c4a6e" />
            <stop offset="100%" stopColor="#082f49" />
          </linearGradient>
        </defs>

        <path d={aPath} fill="none" stroke="#f0abfc" strokeOpacity={0.4} strokeWidth={1.6} strokeDasharray="5 7" className="fx-march" />
        <path d={sPath} fill="none" stroke="#38bdf8" strokeOpacity={0.4} strokeWidth={1.6} strokeDasharray="5 7" className="fx-march" />
        <text x={410} y={24} textAnchor="middle" fontSize={11} fill="#f0abfc" style={{ fontFamily: "var(--font-mono)" }}>
          action aₜ {d ? `= ${d.kind === "maintain" ? "MAINT" : "DISPATCH"} M${d.machine}` : ""}
        </text>
        <text x={410} y={218} textAnchor="middle" fontSize={11} fill="#7dd3fc" style={{ fontFamily: "var(--font-mono)" }}>
          state sₜ₊₁ · risk scores  +  reward rₜ {d ? `= ${d.reward.toFixed(2)}` : ""}
        </text>

        {d && (
          <g key={key}>
            <g>
              <circle r={7} fill="#f0abfc" style={{ filter: "drop-shadow(0 0 8px #f0abfc)" }}>
                <animateMotion dur="1.1s" fill="freeze" keyPoints="0;1" keyTimes="0;1" calcMode="spline" keySplines="0.4 0 0.2 1">
                  <mpath href="#aPath" />
                </animateMotion>
                <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.85;1" dur="1.1s" fill="freeze" />
              </circle>
            </g>
            <g>
              <circle r={6} fill="#38bdf8" style={{ filter: "drop-shadow(0 0 8px #38bdf8)" }}>
                <animateMotion dur="1.1s" begin="1.1s" fill="freeze" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.4 0 0.2 1">
                  <mpath href="#sPath" />
                </animateMotion>
                <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.01;0.85;1" dur="1.1s" begin="1.1s" fill="freeze" />
              </circle>
              <circle r={5} fill={d.reward >= 0 ? "#34d399" : "#fb7185"} style={{ filter: "drop-shadow(0 0 8px currentColor)" }}>
                <animateMotion dur="1.1s" begin="1.25s" fill="freeze" calcMode="spline" keyPoints="0;1" keyTimes="0;1" keySplines="0.4 0 0.2 1">
                  <mpath href="#sPath" />
                </animateMotion>
                <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.01;0.85;1" dur="1.1s" begin="1.25s" fill="freeze" />
              </circle>
            </g>
          </g>
        )}

        {/* agent */}
        <g transform="translate(40,35)">
          <rect width={210} height={150} rx={22} fill="url(#agentGrad)" stroke="#c084fc" strokeOpacity={0.6} />
          <circle cx={105} cy={60} r={30} fill="none" stroke="#e879f9" strokeOpacity={0.5} className="fx-ring" style={{ transformOrigin: "105px 60px", transformBox: "view-box" }} />
          <g transform="translate(105,60)">
            {[0, 1, 2, 3, 4, 5].map((i) => {
              const a = (i / 6) * Math.PI * 2;
              return <circle key={i} cx={Math.cos(a) * 20} cy={Math.sin(a) * 20} r={4} fill="#f0abfc" className="pulse-dot" style={{ animationDelay: `${i * 0.2}s`, transformOrigin: `${Math.cos(a) * 20}px ${Math.sin(a) * 20}px`, transformBox: "view-box" }} />;
            })}
            <circle r={9} fill="#fae8ff" />
            {[0, 1, 2, 3, 4, 5].map((i) => {
              const a = (i / 6) * Math.PI * 2;
              return <line key={`l${i}`} x1={0} y1={0} x2={Math.cos(a) * 20} y2={Math.sin(a) * 20} stroke="#f0abfc" strokeOpacity={0.5} />;
            })}
          </g>
          <text x={105} y={115} textAnchor="middle" fontSize={15} fontWeight={700} fill="#fae8ff" style={{ fontFamily: "var(--font-display)" }}>PPO Agent</text>
          <text x={105} y={133} textAnchor="middle" fontSize={9.5} fill="#c4b5fd" style={{ fontFamily: "var(--font-mono)" }}>π_θ(a | s) · V_φ(s)</text>
        </g>

        {/* environment */}
        <g transform="translate(570,35)">
          <rect width={210} height={150} rx={22} fill="url(#envGrad)" stroke="#38bdf8" strokeOpacity={0.6} />
          <g transform="translate(105,58)">
            <path d="M-22 -4 h44 v22 h-44 z" fill="#0e7490" stroke="#67e8f9" />
            <path d="M-18 -4 l0 -16 l10 8 l0 -8 l10 8 l0 -8 l10 8 l0 16 z" fill="#0e7490" stroke="#67e8f9" />
            <rect x={-16} y={5} width={7} height={8} fill="#cffafe" className="fx-blink" />
            <rect x={-3} y={5} width={7} height={8} fill="#cffafe" className="fx-blink" style={{ animationDelay: ".4s" }} />
            <rect x={10} y={5} width={7} height={8} fill="#cffafe" className="fx-blink" style={{ animationDelay: ".8s" }} />
            {[0, 1, 2].map((i) => (
              <circle key={i} className="fx-smoke" cx={-8 + i * 10} cy={-22} r={3.5} fill="#94a3b8" style={{ animationDelay: `${i * 0.7}s` }} />
            ))}
          </g>
          <text x={105} y={115} textAnchor="middle" fontSize={15} fontWeight={700} fill="#e0f2fe" style={{ fontFamily: "var(--font-display)" }}>Job-Shop Env</text>
          <text x={105} y={133} textAnchor="middle" fontSize={9.5} fill="#7dd3fc" style={{ fontFamily: "var(--font-mono)" }}>machines · hazard · C-MAPSS risk</text>
        </g>
      </svg>
    </div>
  );
}
