"use client";

import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { ParsedEpisode } from "@/types/eventLog";
import { getMachineRiskAtTime, getMachineStatusAtTime } from "@/lib/parseEpisode";

interface RiskHeatmapProps {
  episode: ParsedEpisode;
  currentTime: number;
  onScrub: (time: number) => void;
}

export function getRiskBand(risk: number): {
  label: "NOMINAL" | "DEGRADING" | "CRITICAL";
  badgeClass: string;
  barColor: string;
  glowColor: string;
} {
  if (risk >= 0.7) {
    return {
      label: "CRITICAL",
      badgeClass:
        "border-rose-500/50 bg-rose-950/70 text-rose-300 shadow-[0_0_10px_rgba(244,63,94,0.25)]",
      barColor: "#f43f5e",
      glowColor: "rgba(244, 63, 94, 0.45)",
    };
  }
  if (risk >= 0.3) {
    return {
      label: "DEGRADING",
      badgeClass: "border-amber-500/45 bg-amber-950/60 text-amber-300",
      barColor: "#f59e0b",
      glowColor: "rgba(245, 158, 11, 0.35)",
    };
  }
  return {
    label: "NOMINAL",
    badgeClass: "border-emerald-500/40 bg-emerald-950/50 text-emerald-300",
    barColor: "#10b981",
    glowColor: "rgba(16, 185, 129, 0.3)",
  };
}

export default function RiskHeatmap({
  episode,
  currentTime,
  onScrub,
}: RiskHeatmapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState<number>(960);
  const [hoverCell, setHoverCell] = useState<{
    machine: number;
    t: number;
    risk: number;
  } | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0) {
          setWidth(entry.contentRect.width);
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const nMachines = episode.meta.n_machines;
  const rowHeight = 38;
  // Exact same horizontal margins as GanttChart so time axes align pixel-for-pixel
  const margin = { top: 22, right: 24, bottom: 32, left: 152 };
  const height = margin.top + margin.bottom + nMachines * rowHeight;

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const innerWidth = Math.max(260, width - margin.left - margin.right - 32);
    const innerHeight = nMachines * rowHeight;

    const xScale = d3
      .scaleLinear()
      .domain([0, episode.makespan])
      .range([0, innerWidth]);

    const machineDomain = d3.range(nMachines).map((m) => `M${m}`);
    const yScale = d3
      .scaleBand<string>()
      .domain(machineDomain)
      .range([0, innerHeight])
      .padding(0.18);

    // Rich thermal color scale: deep emerald -> teal -> amber -> crimson
    const colorScale = d3.scaleSequential<string>((t) =>
      d3.interpolateRgbBasis([
        "#064e3b",
        "#059669",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#991b1b",
      ])(Math.max(0, Math.min(1, t)))
    );

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const bandHeight = yScale.bandwidth();

    // Left Machine Labels with live risk bar
    for (let m = 0; m < nMachines; m++) {
      const yPos = yScale(`M${m}`) ?? 0;
      const liveRisk = getMachineRiskAtTime(episode, m, currentTime);
      const band = getRiskBand(liveRisk);

      const labelG = g
        .append("g")
        .attr("transform", `translate(${-margin.left + 4}, ${yPos})`);

      labelG
        .append("rect")
        .attr("width", margin.left - 14)
        .attr("height", bandHeight)
        .attr("rx", 6)
        .attr("fill", "rgba(15, 23, 42, 0.85)")
        .attr("stroke", "rgba(148, 163, 184, 0.14)");

      labelG
        .append("text")
        .attr("x", 12)
        .attr("y", bandHeight / 2 + 4)
        .attr("fill", "#e2e8f0")
        .attr("font-size", "11px")
        .attr("font-weight", "700")
        .text(`M0${m}`);

      labelG
        .append("text")
        .attr("x", margin.left - 24)
        .attr("y", bandHeight / 2 + 4)
        .attr("text-anchor", "end")
        .attr("fill", band.barColor)
        .attr("font-family", "var(--font-mono), monospace")
        .attr("font-size", "11px")
        .attr("font-weight", "700")
        .text(`${(liveRisk * 100).toFixed(0)}%`);
    }

    // Discretize timeline into fine 2-unit thermal slices
    const step = 2;
    const cells: {
      machine: number;
      t0: number;
      t1: number;
      risk: number;
    }[] = [];

    for (let m = 0; m < nMachines; m++) {
      for (let t = 0; t < episode.makespan; t += step) {
        const tMid = Math.min(episode.makespan, t + step / 2);
        const risk = getMachineRiskAtTime(episode, m, tMid);
        cells.push({
          machine: m,
          t0: t,
          t1: Math.min(episode.makespan, t + step),
          risk,
        });
      }
    }

    g.selectAll(".heat-cell")
      .data(cells)
      .join("rect")
      .attr("class", "heat-cell")
      .attr("x", (d) => xScale(d.t0))
      .attr("y", (d) => yScale(`M${d.machine}`) ?? 0)
      .attr("width", (d) => Math.max(1, xScale(d.t1) - xScale(d.t0) - 0.6))
      .attr("height", bandHeight)
      .attr("rx", 2.5)
      .attr("fill", (d) => colorScale(d.risk))
      .attr("opacity", (d) => (d.t0 > currentTime ? 0.34 : 0.92))
      .style("cursor", "pointer")
      .on("mousemove", (_, d) => {
        setHoverCell({ machine: d.machine, t: d.t0, risk: d.risk });
      })
      .on("mouseleave", () => setHoverCell(null))
      .on("click", (_, d) => onScrub(d.t0));

    // Overlay a subtle horizon sparkline curve inside each machine row
    for (let m = 0; m < nMachines; m++) {
      const yTop = yScale(`M${m}`) ?? 0;
      const pts: [number, number][] = [];
      for (let t = 0; t <= episode.makespan; t += 2) {
        const r = getMachineRiskAtTime(episode, m, t);
        pts.push([xScale(t), yTop + bandHeight - r * (bandHeight - 4) - 2]);
      }
      const lineGen = d3.line<[number, number]>().curve(d3.curveMonotoneX);
      g.append("path")
        .attr("d", lineGen(pts) ?? "")
        .attr("fill", "none")
        .attr("stroke", "rgba(255, 255, 255, 0.55)")
        .attr("stroke-width", 1.25)
        .style("pointer-events", "none");
    }

    // X-Axis Bottom
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(11)
      .tickSize(5)
      .tickFormat((d) => `${d}u`);
    const xAxisGroup = g
      .append("g")
      .attr("transform", `translate(0,${innerHeight + 2})`)
      .call(xAxis);
    xAxisGroup.select(".domain").attr("stroke", "rgba(148,163,184,0.2)");
    xAxisGroup.selectAll("line").attr("stroke", "rgba(148,163,184,0.2)");
    xAxisGroup
      .selectAll("text")
      .attr("fill", "#94a3b8")
      .attr("font-family", "var(--font-mono), monospace")
      .attr("font-size", "11px");

    // Synchronized Laser Playhead
    const playheadX = xScale(
      Math.max(0, Math.min(episode.makespan, currentTime))
    );
    g.append("line")
      .attr("x1", playheadX)
      .attr("x2", playheadX)
      .attr("y1", -6)
      .attr("y2", innerHeight + 4)
      .attr("stroke", "#38bdf8")
      .attr("stroke-width", 2.5)
      .style("pointer-events", "none");
  }, [episode, currentTime, width, nMachines, onScrub]);

  return (
    <div
      ref={containerRef}
      className="rajs-panel w-full select-none rounded-2xl p-5"
    >
      {/* Header + Thermal Scale Legend */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-amber-400 shadow-[0_0_8px_#f59e0b]" />
            <h2 className="font-display text-base font-bold tracking-tight text-white sm:text-lg">
              C-MAPSS Calibrated Machine Risk Telemetry &amp; Horizon Heatmap
            </h2>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Layer 1 (OC-SVM / IForest) + Layer 2 Isotonic Regression risk score ∈ [0, 1] with overlaid degradation waveforms.
          </p>
        </div>

        {/* Thermal Legend & Hover Readout */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {hoverCell && (
            <span className="rounded-lg border border-cyan-500/40 bg-cyan-950/60 px-2.5 py-1 font-mono text-[11px] font-semibold text-cyan-300">
              M0{hoverCell.machine} @ t={hoverCell.t}u → Risk{" "}
              {(hoverCell.risk * 100).toFixed(0)}% ({hoverCell.risk.toFixed(2)})
            </span>
          )}

          <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-1.5">
            <span className="font-mono text-[10px] text-emerald-400">0.0</span>
            <div className="h-2 w-28 rounded-full bg-gradient-to-r from-emerald-700 via-amber-500 to-rose-600" />
            <span className="font-mono text-[10px] text-rose-400">1.0</span>
          </div>
        </div>
      </div>

      {/* Live 6-Machine Telemetry Gauge Cards at currentTime */}
      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {Array.from({ length: nMachines }, (_, m) => {
          const risk = getMachineRiskAtTime(episode, m, currentTime);
          const band = getRiskBand(risk);
          const status = getMachineStatusAtTime(episode, m, currentTime);
          const activeSegments = Math.round(risk * 10);

          return (
            <div
              key={m}
              className="rajs-card-inner relative overflow-hidden rounded-xl p-3 transition hover:border-slate-700"
            >
              {/* Top status bar accent */}
              <div
                className="absolute inset-x-0 top-0 h-[2px]"
                style={{ backgroundColor: band.barColor }}
              />

              <div className="flex items-center justify-between">
                <span className="font-display text-xs font-bold tracking-wide text-slate-200">
                  MACHINE 0{m}
                </span>
                <span
                  className={`rounded-md border px-1.5 py-0.5 font-mono text-[9px] font-bold tracking-wider ${band.badgeClass}`}
                >
                  {band.label}
                </span>
              </div>

              <div className="mt-2 flex items-baseline justify-between">
                <span
                  className="font-mono text-2xl font-bold tracking-tight"
                  style={{ color: band.barColor }}
                >
                  {(risk * 100).toFixed(0)}%
                </span>
                <span className="font-mono text-[11px] text-slate-400">
                  r = {risk.toFixed(2)}
                </span>
              </div>

              {/* 10-Segment Industrial LED Bar */}
              <div className="mt-2 grid grid-cols-10 gap-1">
                {Array.from({ length: 10 }, (__, segIdx) => {
                  const lit = segIdx < activeSegments;
                  const segColor =
                    segIdx >= 7
                      ? "#f43f5e"
                      : segIdx >= 3
                      ? "#f59e0b"
                      : "#10b981";
                  return (
                    <div
                      key={segIdx}
                      className="h-1.5 rounded-xs transition-all duration-150"
                      style={{
                        backgroundColor: lit ? segColor : "rgba(30, 41, 59, 0.8)",
                        boxShadow: lit ? `0 0 6px ${segColor}88` : "none",
                      }}
                    />
                  );
                })}
              </div>

              <div className="mt-2 flex items-center justify-between text-[11px]">
                <span className="truncate font-medium text-slate-300">
                  {status.detail}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Synchronized D3 Heatmap + Waveform Matrix */}
      <div className="overflow-x-auto">
        <svg
          ref={svgRef}
          width={Math.max(600, width - 32)}
          height={height}
          className="overflow-visible"
        />
      </div>
    </div>
  );
}
