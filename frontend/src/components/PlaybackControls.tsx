"use client";

import React from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepBack,
  StepForward,
  Wrench,
  AlertTriangle,
  Sparkles,
} from "lucide-react";
import { ParsedEpisode } from "@/types/eventLog";

interface PlaybackControlsProps {
  episode: ParsedEpisode;
  currentTime: number;
  makespan: number;
  isPlaying: boolean;
  speed: number;
  onTogglePlay: () => void;
  onReset: () => void;
  onStep: (delta: number) => void;
  onScrub: (time: number) => void;
  onChangeSpeed: (speed: number) => void;
}

const SPEEDS = [0.5, 1, 2, 4];

export default function PlaybackControls({
  episode,
  currentTime,
  makespan,
  isPlaying,
  speed,
  onTogglePlay,
  onReset,
  onStep,
  onScrub,
  onChangeSpeed,
}: PlaybackControlsProps) {
  const progressPct = makespan > 0 ? Math.min(100, Math.max(0, (currentTime / makespan) * 100)) : 0;

  return (
    <div className="rajs-panel relative overflow-hidden rounded-2xl p-4 sm:px-6 sm:py-4">
      {/* Subtle top cyan accent line */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent" />

      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Left: Tactile Transport Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={onReset}
            title="Reset timeline to t=0"
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-700/80 bg-slate-900/90 text-slate-300 shadow-inner transition hover:border-slate-500 hover:bg-slate-800 hover:text-white active:scale-95"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          <button
            onClick={() => onStep(-2)}
            title="Step backward (-2u)"
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-700/80 bg-slate-900/90 text-slate-300 shadow-inner transition hover:border-slate-500 hover:bg-slate-800 hover:text-white active:scale-95"
          >
            <StepBack className="h-4 w-4" />
          </button>

          <button
            onClick={onTogglePlay}
            className={`flex h-10 items-center gap-2.5 rounded-xl px-5 text-sm font-semibold tracking-wide text-white shadow-lg transition active:scale-95 ${
              isPlaying
                ? "bg-gradient-to-r from-amber-500 to-orange-500 shadow-amber-500/25 hover:from-amber-400 hover:to-orange-400"
                : "bg-gradient-to-r from-cyan-500 to-blue-600 shadow-cyan-500/25 hover:from-cyan-400 hover:to-blue-500"
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="h-4 w-4 fill-current" />
                <span>Pause Simulation</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-current" />
                <span>{currentTime >= makespan ? "Replay Episode" : "Play Simulation"}</span>
              </>
            )}
          </button>

          <button
            onClick={() => onStep(2)}
            title="Step forward (+2u)"
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-700/80 bg-slate-900/90 text-slate-300 shadow-inner transition hover:border-slate-500 hover:bg-slate-800 hover:text-white active:scale-95"
          >
            <StepForward className="h-4 w-4" />
          </button>
        </div>

        {/* Center: Dynamic Event Bookmarks from Episode */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-mono text-[11px] uppercase tracking-wider text-slate-400">
            Key Events:
          </span>

          {episode.breakdowns.map((bd) => (
            <button
              key={bd.id}
              onClick={() => onScrub(bd.breakdownTime)}
              className="group flex items-center gap-1.5 rounded-lg border border-rose-500/35 bg-rose-950/40 px-2.5 py-1.5 font-medium text-rose-300 transition hover:border-rose-400/70 hover:bg-rose-900/50 hover:text-rose-100"
            >
              <AlertTriangle className="h-3.5 w-3.5 text-rose-400 transition group-hover:scale-110" />
              <span>M{bd.machine} Breakdown</span>
              <span className="rounded bg-rose-500/20 px-1.5 py-0.5 font-mono text-[10px] text-rose-200">
                t={bd.breakdownTime}
              </span>
            </button>
          ))}

          {episode.maintenances.map((pm) => (
            <button
              key={pm.id}
              onClick={() => onScrub(pm.start)}
              className="group flex items-center gap-1.5 rounded-lg border border-emerald-500/35 bg-emerald-950/40 px-2.5 py-1.5 font-medium text-emerald-300 transition hover:border-emerald-400/70 hover:bg-emerald-900/50 hover:text-emerald-100"
            >
              <Wrench className="h-3.5 w-3.5 text-emerald-400 transition group-hover:scale-110" />
              <span>M{pm.machine} Preventive PM</span>
              <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[10px] text-emerald-200">
                t={pm.start}
              </span>
            </button>
          ))}

          <button
            onClick={() => onScrub(makespan)}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-900/80 px-2.5 py-1.5 font-medium text-slate-300 transition hover:border-cyan-500/50 hover:text-white"
          >
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span>Makespan</span>
            <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
              t={makespan}
            </span>
          </button>
        </div>

        {/* Right: Speed Selector & Digital Clock Readout */}
        <div className="flex items-center gap-3">
          <div className="flex items-center rounded-xl border border-slate-800 bg-slate-950/90 p-1 text-xs">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => onChangeSpeed(s)}
                className={`rounded-lg px-2.5 py-1 font-mono font-semibold transition ${
                  speed === s
                    ? "bg-gradient-to-r from-cyan-500/25 to-blue-500/25 text-cyan-300 border border-cyan-500/40 shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {s}×
              </button>
            ))}
          </div>

          <div className="flex items-baseline gap-1.5 rounded-xl border border-cyan-500/30 bg-slate-950/95 px-3.5 py-1.5 font-mono shadow-inner">
            <span className="text-[11px] uppercase tracking-wider text-slate-400">T</span>
            <span className="text-lg font-bold text-cyan-400">
              {currentTime.toFixed(1)}
            </span>
            <span className="text-xs text-slate-500">/ {makespan}u</span>
          </div>
        </div>
      </div>

      {/* Custom Precision Timeline Track */}
      <div className="mt-4">
        <div className="relative flex h-8 items-center select-none">
          {/* Base Track */}
          <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-slate-950 border border-slate-800/90 shadow-inner">
            {/* Active Glowing Progress Fill */}
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-sky-400 to-blue-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>

          {/* Breakdown & Maintenance Markers on the Scrubber Track */}
          {episode.breakdowns.map((bd) => {
            const leftPct = (bd.breakdownTime / Math.max(1, makespan)) * 100;
            return (
              <div
                key={`pin-${bd.id}`}
                style={{ left: `${leftPct}%` }}
                className="pointer-events-none absolute -top-1 -translate-x-1/2"
              >
                <div className="h-4 w-1 rounded-full bg-rose-500 shadow-[0_0_8px_#f43f5e]" />
              </div>
            );
          })}

          {episode.maintenances.map((pm) => {
            const leftPct = (pm.start / Math.max(1, makespan)) * 100;
            return (
              <div
                key={`pin-${pm.id}`}
                style={{ left: `${leftPct}%` }}
                className="pointer-events-none absolute -top-1 -translate-x-1/2"
              >
                <div className="h-4 w-1 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
              </div>
            );
          })}

          {/* Glowing Playhead Thumb */}
          <div
            style={{ left: `${progressPct}%` }}
            className="pointer-events-none absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
          >
            <div className="h-5 w-5 rounded-full border-2 border-white bg-cyan-400 shadow-[0_0_15px_rgba(56,189,248,0.9)]" />
          </div>

          {/* Interactive Range Input Overlay */}
          <input
            type="range"
            min={0}
            max={makespan}
            step={0.5}
            value={currentTime}
            onChange={(e) => onScrub(Number(e.target.value))}
            aria-label="Simulation time scrubber"
            className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          />
        </div>

        {/* Scale Ticks under Scrubber */}
        <div className="mt-0.5 flex justify-between px-1 font-mono text-[10px] text-slate-500">
          <span>t = 0u</span>
          <span>t = {Math.round(makespan * 0.25)}u</span>
          <span>t = {Math.round(makespan * 0.5)}u</span>
          <span>t = {Math.round(makespan * 0.75)}u</span>
          <span>t = {makespan}u (Makespan)</span>
        </div>
      </div>
    </div>
  );
}
