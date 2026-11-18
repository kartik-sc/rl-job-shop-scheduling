"use client";

import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { OperationInterval, ParsedEpisode } from "@/types/eventLog";
import { getMachineRiskAtTime, getMachineStatusAtTime } from "@/lib/parseEpisode";

export const JOB_PALETTE = [
  { id: 0, main: "#38bdf8", dark: "#0369a1", border: "#7dd3fc", label: "Job 0" },
  { id: 1, main: "#a855f7", dark: "#6b21a8", border: "#d8b4fe", label: "Job 1" },
  { id: 2, main: "#14b8a6", dark: "#0f766e", border: "#5eead4", label: "Job 2" },
  { id: 3, main: "#f59e0b", dark: "#b45309", border: "#fcd34d", label: "Job 3" },
  { id: 4, main: "#6366f1", dark: "#3730a3", border: "#a5b4fc", label: "Job 4" },
  { id: 5, main: "#ec4899", dark: "#9d174d", border: "#f9a8d4", label: "Job 5" },
];

export const JOB_COLORS = JOB_PALETTE.map((p) => p.main);

interface GanttChartProps {
  episode: ParsedEpisode;
  currentTime: number;
  onScrub: (time: number) => void;
  selectedJob: number | null;
  onSelectJob: (job: number | null) => void;
}

interface TooltipState {
  x: number;
  y: number;
  title: string;
  subtitle: string;
  metaLine: string;
  badgeColor: string;
}

export default function GanttChart({
  episode,
  currentTime,
  onScrub,
  selectedJob,
  onSelectJob,
}: GanttChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState<number>(960);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [hoveredJob, setHoveredJob] = useState<number | null>(null);

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

  const rowHeight = 56;
  const margin = { top: 34, right: 24, bottom: 36, left: 152 };
  const nMachines = episode.meta.n_machines;
  const height = margin.top + margin.bottom + nMachines * rowHeight;

  const activeHighlightJob = selectedJob !== null ? selectedJob : hoveredJob;

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
      .padding(0.22);

    // SVG Definitions: Job Gradients, Hatch Patterns & Glow Filter
    const defs = svg.append("defs");

    JOB_PALETTE.forEach((p, idx) => {
      const grad = defs
        .append("linearGradient")
        .attr("id", `job-grad-${idx}`)
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "0%")
        .attr("y2", "100%");
      grad.append("stop").attr("offset", "0%").attr("stop-color", p.main).attr("stop-opacity", 0.95);
      grad.append("stop").attr("offset", "100%").attr("stop-color", p.dark).attr("stop-opacity", 0.95);
    });

    // Preventive Maintenance Emerald Hatch
    const maintPattern = defs
      .append("pattern")
      .attr("id", "pm-hatch")
      .attr("width", 10)
      .attr("height", 10)
      .attr("patternUnits", "userSpaceOnUse")
      .attr("patternTransform", "rotate(45)");
    maintPattern
      .append("rect")
      .attr("width", 10)
      .attr("height", 10)
      .attr("fill", "#022c22");
    maintPattern
      .append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", 10)
      .attr("stroke", "#10b981")
      .attr("stroke-width", 3.5)
      .attr("stroke-opacity", 0.65);

    // Breakdown Repair Crimson Hatch
    const bdPattern = defs
      .append("pattern")
      .attr("id", "bd-hatch")
      .attr("width", 10)
      .attr("height", 10)
      .attr("patternUnits", "userSpaceOnUse")
      .attr("patternTransform", "rotate(-45)");
    bdPattern
      .append("rect")
      .attr("width", 10)
      .attr("height", 10)
      .attr("fill", "#450a0a");
    bdPattern
      .append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", 10)
      .attr("stroke", "#f43f5e")
      .attr("stroke-width", 3.5)
      .attr("stroke-opacity", 0.7);

    // Glow filter for playhead
    const glow = defs
      .append("filter")
      .attr("id", "laser-glow")
      .attr("x", "-50%")
      .attr("y", "-20%")
      .attr("width", "200%")
      .attr("height", "140%");
    glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const feMerge = glow.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "blur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // 1. Machine Lanes Background + Left Telemetry Labels
    for (let m = 0; m < nMachines; m++) {
      const yTop = m * rowHeight;
      const status = getMachineStatusAtTime(episode, m, currentTime);
      const risk = getMachineRiskAtTime(episode, m, currentTime);

      // Compute machine productive utilization %
      const busyTime = episode.operations
        .filter((o) => o.machine === m && !o.interrupted)
        .reduce((acc, o) => acc + (o.end - o.start), 0);
      const utilPct = Math.round((busyTime / Math.max(1, episode.makespan)) * 100);

      // Lane background
      g.append("rect")
        .attr("x", 0)
        .attr("y", yTop + 3)
        .attr("width", innerWidth)
        .attr("height", rowHeight - 6)
        .attr("rx", 8)
        .attr("fill", m % 2 === 0 ? "rgba(15, 23, 42, 0.55)" : "rgba(15, 23, 42, 0.3)")
        .attr("stroke", "rgba(148, 163, 184, 0.08)")
        .attr("stroke-width", 1);

      // Left Machine Label Card inside SVG margin
      const labelG = g
        .append("g")
        .attr("transform", `translate(${-margin.left + 4}, ${yTop + 5})`);

      labelG
        .append("rect")
        .attr("width", margin.left - 14)
        .attr("height", rowHeight - 10)
        .attr("rx", 8)
        .attr("fill", "rgba(15, 23, 42, 0.85)")
        .attr(
          "stroke",
          status.state === "breakdown"
            ? "rgba(244, 63, 94, 0.5)"
            : status.state === "maintenance"
            ? "rgba(16, 185, 129, 0.5)"
            : "rgba(148, 163, 184, 0.14)"
        )
        .attr("stroke-width", 1);

      // Live Status Dot
      const dotColor =
        status.state === "breakdown"
          ? "#f43f5e"
          : status.state === "maintenance"
          ? "#10b981"
          : status.state === "operating"
          ? "#38bdf8"
          : "#475569";

      labelG
        .append("circle")
        .attr("cx", 14)
        .attr("cy", 16)
        .attr("r", 4)
        .attr("fill", dotColor);

      labelG
        .append("text")
        .attr("x", 24)
        .attr("y", 20)
        .attr("fill", "#f1f5f9")
        .attr("font-size", "12px")
        .attr("font-weight", "700")
        .text(`MACHINE 0${m}`);

      labelG
        .append("text")
        .attr("x", 14)
        .attr("y", 36)
        .attr("fill", "#94a3b8")
        .attr("font-family", "var(--font-mono), monospace")
        .attr("font-size", "10px")
        .text(`Util ${utilPct}% · r=${risk.toFixed(2)}`);
    }

    // 2. Vertical Time Grid Lines
    const xTicks = xScale.ticks(11);
    g.selectAll(".x-grid")
      .data(xTicks)
      .join("line")
      .attr("x1", (d) => xScale(d))
      .attr("x2", (d) => xScale(d))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "rgba(148, 163, 184, 0.09)")
      .attr("stroke-dasharray", "3,4");

    // Top & Bottom X-Axis Labels
    const xAxisBottom = d3
      .axisBottom(xScale)
      .ticks(11)
      .tickSize(5)
      .tickFormat((d) => `${d}u`);
    const bottomAxisG = g
      .append("g")
      .attr("transform", `translate(0,${innerHeight + 2})`)
      .call(xAxisBottom);
    bottomAxisG.select(".domain").attr("stroke", "rgba(148,163,184,0.2)");
    bottomAxisG.selectAll("line").attr("stroke", "rgba(148,163,184,0.2)");
    bottomAxisG
      .selectAll("text")
      .attr("fill", "#94a3b8")
      .attr("font-family", "var(--font-mono), monospace")
      .attr("font-size", "11px");

    const barHeight = yScale.bandwidth();

    // 3. Optional Job Precedence Routing Curves when a Job is highlighted
    if (activeHighlightJob !== null) {
      const jobOps = episode.operations
        .filter((o) => o.job === activeHighlightJob)
        .sort((a, b) => a.start - b.start);

      const routeColor =
        JOB_PALETTE[activeHighlightJob % JOB_PALETTE.length].border;

      for (let i = 0; i < jobOps.length - 1; i++) {
        const curr = jobOps[i];
        const next = jobOps[i + 1];
        const x1 = xScale(curr.end);
        const y1 = (yScale(`M${curr.machine}`) ?? 0) + barHeight / 2;
        const x2 = xScale(next.start);
        const y2 = (yScale(`M${next.machine}`) ?? 0) + barHeight / 2;

        const pathData = d3.linkHorizontal<
          unknown,
          { x: number; y: number }
        >()({
          source: { x: x1, y: y1 },
          target: { x: x2, y: y2 },
        });

        if (pathData) {
          g.append("path")
            .attr("d", pathData)
            .attr("fill", "none")
            .attr("stroke", routeColor)
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "5,4")
            .attr("opacity", 0.85)
            .style("pointer-events", "none");
        }
      }
    }

    // 4. Operation Bars
    const opGroups = g
      .selectAll<SVGGElement, OperationInterval>(".op-group")
      .data(episode.operations)
      .join("g")
      .attr("class", "op-group")
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        onSelectJob(selectedJob === d.job ? null : d.job);
      })
      .on("mouseenter", (_, d) => setHoveredJob(d.job))
      .on("mousemove", (event, d) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const duration = d.end - d.start;
        const status = d.interrupted
          ? `Interrupted by Breakdown at t=${d.end}u`
          : currentTime >= d.end
          ? "Completed"
          : currentTime >= d.start
          ? "Active Now"
          : "Scheduled";
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
          title: `Job ${d.job} · Operation ${d.op}`,
          subtitle: `Machine 0${d.machine} · Window: t=${d.start}u → ${d.end}u (${duration}u)`,
          metaLine: status,
          badgeColor: JOB_PALETTE[d.job % JOB_PALETTE.length].main,
        });
      })
      .on("mouseleave", () => {
        setHoveredJob(null);
        setTooltip(null);
      });

    opGroups
      .append("rect")
      .attr("x", (d) => xScale(d.start))
      .attr("y", (d) => yScale(`M${d.machine}`) ?? 0)
      .attr("width", (d) => Math.max(4, xScale(d.end) - xScale(d.start)))
      .attr("height", barHeight)
      .attr("rx", 7)
      .attr("fill", (d) => `url(#job-grad-${d.job % JOB_PALETTE.length})`)
      .attr("opacity", (d) => {
        if (activeHighlightJob !== null && activeHighlightJob !== d.job) {
          return 0.18;
        }
        if (d.start > currentTime) return 0.38;
        return d.interrupted ? 0.72 : 1;
      })
      .attr("stroke", (d) => {
        if (d.interrupted) return "#fb7185";
        if (currentTime >= d.start && currentTime < d.end) return "#ffffff";
        return JOB_PALETTE[d.job % JOB_PALETTE.length].border;
      })
      .attr("stroke-width", (d) =>
        d.interrupted ||
        (currentTime >= d.start && currentTime < d.end) ||
        activeHighlightJob === d.job
          ? 2
          : 1
      )
      .attr("stroke-dasharray", (d) => (d.interrupted ? "4,2" : "none"));

    // Operation Primary Label (J0·O1)
    opGroups
      .append("text")
      .attr("x", (d) => xScale(d.start) + (xScale(d.end) - xScale(d.start)) / 2)
      .attr("y", (d) => (yScale(`M${d.machine}`) ?? 0) + barHeight / 2 + 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#ffffff")
      .attr("font-family", "var(--font-mono), monospace")
      .attr("font-size", "11px")
      .attr("font-weight", "700")
      .style("pointer-events", "none")
      .attr("opacity", (d) => {
        const w = xScale(d.end) - xScale(d.start);
        if (w < 36) return 0;
        if (activeHighlightJob !== null && activeHighlightJob !== d.job) {
          return 0.2;
        }
        return d.start > currentTime ? 0.55 : 1;
      })
      .text((d) =>
        d.interrupted ? `J${d.job}·O${d.op} ⚡` : `J${d.job}·O${d.op}`
      );

    // 5. Preventive Maintenance Blocks
    const pmGroups = g
      .selectAll(".pm-group")
      .data(episode.maintenances)
      .join("g")
      .attr("class", "pm-group")
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
          title: `Preventive Maintenance (Machine 0${d.machine})`,
          subtitle: `Window: t=${d.start}u → ${d.end}u (t_pm = ${d.end - d.start}u)`,
          metaLine: `Engine Reset: Risk ${(d.riskBefore ?? 0).toFixed(2)} → ${(
            d.riskAfter ?? 0
          ).toFixed(2)}`,
          badgeColor: "#10b981",
        });
      })
      .on("mouseleave", () => setTooltip(null));

    pmGroups
      .append("rect")
      .attr("x", (d) => xScale(d.start))
      .attr("y", (d) => yScale(`M${d.machine}`) ?? 0)
      .attr("width", (d) => Math.max(6, xScale(d.end) - xScale(d.start)))
      .attr("height", barHeight)
      .attr("rx", 7)
      .attr("fill", "url(#pm-hatch)")
      .attr("stroke", "#34d399")
      .attr("stroke-width", 2)
      .attr("opacity", (d) => (d.start > currentTime ? 0.45 : 1));

    pmGroups
      .append("text")
      .attr("x", (d) => xScale(d.start) + (xScale(d.end) - xScale(d.start)) / 2)
      .attr("y", (d) => (yScale(`M${d.machine}`) ?? 0) + barHeight / 2 + 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#d1fae5")
      .attr("font-family", "var(--font-mono), monospace")
      .attr("font-size", "10px")
      .attr("font-weight", "700")
      .style("pointer-events", "none")
      .text("PM");

    // 6. Breakdown & Repair Blocks
    const bdGroups = g
      .selectAll(".bd-group")
      .data(episode.breakdowns)
      .join("g")
      .attr("class", "bd-group")
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
          title: `Unplanned Breakdown · Machine 0${d.machine}`,
          subtitle: `Repair Window: t=${d.breakdownTime}u → ${d.repairEndTime}u (t_repair = ${
            d.repairEndTime - d.breakdownTime
          }u = 5× t_pm)`,
          metaLine: `Interrupted Job ${d.job} · Op ${d.op} at Risk ${(
            d.riskAtBreakdown ?? 0
          ).toFixed(2)} → Restarted at t=${d.repairEndTime}u`,
          badgeColor: "#f43f5e",
        });
      })
      .on("mouseleave", () => setTooltip(null));

    bdGroups
      .append("rect")
      .attr("x", (d) => xScale(d.breakdownTime))
      .attr("y", (d) => yScale(`M${d.machine}`) ?? 0)
      .attr("width", (d) =>
        Math.max(6, xScale(d.repairEndTime) - xScale(d.breakdownTime))
      )
      .attr("height", barHeight)
      .attr("rx", 7)
      .attr("fill", "url(#bd-hatch)")
      .attr("stroke", "#f43f5e")
      .attr("stroke-width", 2)
      .attr("opacity", (d) => (d.breakdownTime > currentTime ? 0.45 : 1));

    bdGroups
      .append("text")
      .attr(
        "x",
        (d) =>
          xScale(d.breakdownTime) +
          (xScale(d.repairEndTime) - xScale(d.breakdownTime)) / 2
      )
      .attr("y", (d) => (yScale(`M${d.machine}`) ?? 0) + barHeight / 2 + 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#ffe4e6")
      .attr("font-family", "var(--font-mono), monospace")
      .attr("font-size", "11px")
      .attr("font-weight", "700")
      .style("pointer-events", "none")
      .text(
        (d) => `⚡ BREAKDOWN REPAIR (${d.repairEndTime - d.breakdownTime}u)`
      );

    // 7. Laser Playhead + Future Veil
    const playheadX = xScale(
      Math.max(0, Math.min(episode.makespan, currentTime))
    );

    g.append("rect")
      .attr("x", playheadX)
      .attr("y", 0)
      .attr("width", Math.max(0, innerWidth - playheadX))
      .attr("height", innerHeight)
      .attr("fill", "#020617")
      .attr("opacity", 0.32)
      .style("pointer-events", "none");

    const playheadG = g.append("g").style("pointer-events", "none");

    playheadG
      .append("line")
      .attr("x1", playheadX)
      .attr("x2", playheadX)
      .attr("y1", -10)
      .attr("y2", innerHeight + 4)
      .attr("stroke", "#38bdf8")
      .attr("stroke-width", 2.5)
      .attr("filter", "url(#laser-glow)");

    playheadG
      .append("rect")
      .attr("x", playheadX - 32)
      .attr("y", -26)
      .attr("width", 64)
      .attr("height", 20)
      .attr("rx", 6)
      .attr("fill", "#0284c7")
      .attr("stroke", "#7dd3fc")
      .attr("stroke-width", 1);

    playheadG
      .append("text")
      .attr("x", playheadX)
      .attr("y", -12)
      .attr("text-anchor", "middle")
      .attr("fill", "#ffffff")
      .attr("font-family", "var(--font-mono), monospace")
      .attr("font-size", "10px")
      .attr("font-weight", "700")
      .text(`t=${currentTime.toFixed(0)}u`);

    // 8. Interactive Click & Drag Scrubbing across the Canvas
    const scrubFromEvent = (sourceEvent: MouseEvent | TouchEvent) => {
      const [mx] = d3.pointer(sourceEvent, g.node());
      const clamped = Math.max(0, Math.min(innerWidth, mx));
      const newTime = Math.round(xScale.invert(clamped) * 2) / 2;
      onScrub(Math.max(0, Math.min(episode.makespan, newTime)));
    };

    const dragBehavior = d3
      .drag<SVGSVGElement, unknown>()
      .on("start.scrub", (event) => {
        const [mx] = d3.pointer(event.sourceEvent, g.node());
        if (mx >= 0 && mx <= innerWidth) {
          scrubFromEvent(event.sourceEvent);
        }
      })
      .on("drag.scrub", (event) => {
        scrubFromEvent(event.sourceEvent);
      });

    svg.call(dragBehavior);
  }, [
    episode,
    currentTime,
    width,
    nMachines,
    onScrub,
    selectedJob,
    onSelectJob,
    activeHighlightJob,
  ]);

  return (
    <div
      ref={containerRef}
      className="rajs-panel relative w-full select-none rounded-2xl p-5"
    >
      {/* Top Section Header & Interactive Job Routing Pills */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#38bdf8]" />
            <h2 className="font-display text-base font-bold tracking-tight text-white sm:text-lg">
              Machine Schedule &amp; Job Routing Gantt
            </h2>
          </div>
          <p className="mt-0.5 text-xs text-slate-400">
            Hover or click any job to trace its routing path across machines. Drag anywhere along the chart to scrub time.
          </p>
        </div>

        {/* Interactive Job Filter & Event Legend */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          {Array.from({ length: episode.meta.n_jobs }, (_, j) => {
            const palette = JOB_PALETTE[j % JOB_PALETTE.length];
            const isSelected = selectedJob === j;
            const isDimmed = selectedJob !== null && selectedJob !== j;
            return (
              <button
                key={j}
                onClick={() => onSelectJob(isSelected ? null : j)}
                onMouseEnter={() => setHoveredJob(j)}
                onMouseLeave={() => setHoveredJob(null)}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 font-medium transition ${
                  isSelected
                    ? "border-cyan-400 bg-slate-800 text-white shadow-[0_0_12px_rgba(56,189,248,0.25)]"
                    : isDimmed
                    ? "border-slate-800/80 bg-slate-950/60 text-slate-500 opacity-50"
                    : "border-slate-800 bg-slate-900/80 text-slate-200 hover:border-slate-600"
                }`}
              >
                <span
                  className="h-2.5 w-2.5 rounded-sm"
                  style={{
                    backgroundColor: palette.main,
                    boxShadow: `0 0 8px ${palette.main}88`,
                  }}
                />
                <span>Job {j}</span>
              </button>
            );
          })}

          <div className="mx-1 h-4 w-[1px] bg-slate-800" />

          <span className="flex items-center gap-1.5 rounded-lg border border-emerald-500/35 bg-emerald-950/40 px-2.5 py-1 font-mono text-[11px] text-emerald-300">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-400" />
            PM (t_pm=8u)
          </span>

          <span className="flex items-center gap-1.5 rounded-lg border border-rose-500/35 bg-rose-950/40 px-2.5 py-1 font-mono text-[11px] text-rose-300">
            <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
            Breakdown (t_repair=40u)
          </span>
        </div>
      </div>

      {/* D3 SVG Canvas */}
      <div className="overflow-x-auto">
        <svg
          ref={svgRef}
          width={Math.max(600, width - 32)}
          height={height}
          className="overflow-visible"
        />
      </div>

      {/* Floating Glass Tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-30 max-w-xs rounded-xl border border-slate-700/90 bg-slate-950/95 px-3.5 py-2.5 text-xs shadow-2xl backdrop-blur-md"
          style={{
            left: Math.min(width - 280, Math.max(16, tooltip.x + 14)),
            top: Math.max(14, tooltip.y - 68),
          }}
        >
          <div className="flex items-center gap-2 font-display text-sm font-bold text-white">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{
                backgroundColor: tooltip.badgeColor,
                boxShadow: `0 0 10px ${tooltip.badgeColor}`,
              }}
            />
            {tooltip.title}
          </div>
          <div className="mt-1 font-mono text-[11px] text-slate-300">
            {tooltip.subtitle}
          </div>
          <div className="mt-1 text-[11px] font-medium text-cyan-300">
            {tooltip.metaLine}
          </div>
        </div>
      )}
    </div>
  );
}
