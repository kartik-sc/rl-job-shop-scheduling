import { EpisodeLog, ParsedEpisode } from "@/types/eventLog";
import { buildEpisodeScene, buildFlows, buildRoutes } from "@/lib/scene";
import { getMachineRiskAtTime, getMachineStatusAtTime } from "@/lib/parseEpisode";
import { mulberry32 } from "@/lib/viz";

/**
 * Illustrative RL view derived from an episode event log.
 * There is no trained policy yet, so action probabilities are a hand-built
 * policy head that agrees with the logged action. State/reward definitions
 * mirror the intended Gymnasium env and should be swapped for real
 * logits/values once the backend logs them.
 */

export interface Decision {
  idx: number;
  t: number;
  kind: "dispatch" | "maintain";
  machine: number;
  job?: number;
  op?: number;
  state: number[]; // [risk x n, busy x n, queue x n]
  probs: number[]; // [WAIT, DISPATCH m x n, MAINT m x n]
  chosen: number;
  reward: number;
  ret: number;
  parts: { time: number; completions: number; breakdown: number; maint: number };
  h1: number[];
  h2: number[];
}

export const H1 = 12;
export const H2 = 8;

export function actionLabels(n: number): string[] {
  return [
    "WAIT",
    ...Array.from({ length: n }, (_, m) => `DISPATCH M${m}`),
    ...Array.from({ length: n }, (_, m) => `MAINT M${m}`),
  ];
}

export function inputLabels(n: number): string[] {
  return [
    ...Array.from({ length: n }, (_, m) => `risk M${m}`),
    ...Array.from({ length: n }, (_, m) => `busy M${m}`),
    ...Array.from({ length: n }, (_, m) => `queue M${m}`),
  ];
}

function softmax(z: number[], temp = 1) {
  const mx = Math.max(...z);
  const e = z.map((v) => Math.exp((v - mx) / temp));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function makeWeights(nIn: number) {
  const rnd = mulberry32(2026);
  const mat = (r: number, c: number) =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => (rnd() - 0.5) * 2));
  return { w1: mat(nIn, H1), w2: mat(H1, H2), w3: mat(H2, 1) };
}

export function forward(x: number[], w: ReturnType<typeof makeWeights>) {
  const h1 = Array.from({ length: H1 }, (_, j) =>
    Math.tanh(x.reduce((a, v, i) => a + v * w.w1[i][j] * 1.4, 0) * 0.9)
  );
  const h2 = Array.from({ length: H2 }, (_, j) =>
    Math.tanh(h1.reduce((a, v, i) => a + v * w.w2[i][j], 0) * 0.8)
  );
  return { h1, h2 };
}

export function deriveDecisions(parsed: ParsedEpisode, log: EpisodeLog): Decision[] {
  const n = parsed.meta.n_machines;
  const routes = buildRoutes(parsed);
  const flows = buildFlows(routes);
  const weights = makeWeights(3 * n);
  const out: Decision[] = [];
  let prevT = 0;
  let ret = 0;
  let evCursor = 0;
  const events = log.events;

  for (const ev of events) {
    const isDispatch = ev.type === "op_start";
    const isMaint = ev.type === "maintenance_start";
    if (!(isDispatch || isMaint) || ev.machine === undefined) continue;
    const t = ev.t;

    let completions = 0;
    let breakdown = 0;
    while (evCursor < events.length && events[evCursor].t <= t) {
      const e = events[evCursor];
      if (e.t > prevT || out.length === 0) {
        if (e.type === "op_end") completions++;
        if (e.type === "breakdown") breakdown++;
      }
      evCursor++;
    }
    const scene = buildEpisodeScene(parsed, routes, flows, Math.max(0, t - 0.01));
    const risk = Array.from({ length: n }, (_, m) => getMachineRiskAtTime(parsed, m, t));
    const busy = Array.from({ length: n }, (_, m) =>
      getMachineStatusAtTime(parsed, m, Math.max(0, t - 0.01)).state === "idle" ? 0 : 1
    );
    const queue = Array.from({ length: n }, (_, m) =>
      Math.min(
        1,
        scene.jobs.filter((j) => j.status === "queued" && j.machine === m).length / 3
      )
    );
    const state = [...risk, ...busy, ...queue];

    const logits = new Array(2 * n + 1).fill(0);
    const anyReady = busy.some((b, m) => b === 0 && queue[m] > 0);
    logits[0] = anyReady ? -0.6 : 0.8;
    for (let m = 0; m < n; m++) {
      logits[1 + m] = (busy[m] ? -2.2 : queue[m] > 0 ? 1.8 : -0.4) - 1.4 * risk[m];
      logits[1 + n + m] = 7 * (risk[m] - 0.62) - 0.5 * queue[m] - (busy[m] ? 0.8 : 0);
    }
    const chosen = isDispatch ? 1 + ev.machine : 1 + n + ev.machine;
    logits[chosen] = Math.max(...logits) + 1.9;
    const probs = softmax(logits);

    const parts = {
      time: -0.04 * Math.max(0, t - prevT),
      completions: completions * 1.0,
      breakdown: breakdown * -6,
      maint: isMaint ? -0.8 : 0,
    };
    const reward = parts.time + parts.completions + parts.breakdown + parts.maint;
    ret += reward;
    const { h1, h2 } = forward(state, weights);

    out.push({
      idx: out.length,
      t,
      kind: isDispatch ? "dispatch" : "maintain",
      machine: ev.machine,
      job: ev.job,
      op: ev.op,
      state,
      probs,
      chosen,
      reward,
      ret,
      parts,
      h1,
      h2,
    });
    prevT = t;
  }
  return out;
}

export interface TrainingCurves {
  iters: number[];
  full: number[];
  blind: number[];
  threshold: number[];
  fullBreak: number[];
  blindBreak: number[];
  thresholdBreak: number[];
}

/** Synthetic, seeded training curves (placeholder until real runs are logged). */
export function syntheticTraining(points = 160): TrainingCurves {
  const rnd = mulberry32(7);
  const g = () => (rnd() + rnd() + rnd() - 1.5) * 2;
  const iters = Array.from({ length: points }, (_, i) => i);
  const full: number[] = [];
  const blind: number[] = [];
  const threshold: number[] = [];
  const fullBreak: number[] = [];
  const blindBreak: number[] = [];
  const thresholdBreak: number[] = [];
  for (const i of iters) {
    const k = 1 - Math.exp(-i / 42);
    const kb = 1 - Math.exp(-i / 30);
    const decay = Math.exp(-i / 60);
    full.push(-96 + 54 * k + g() * (3 + 9 * decay) + (i > 70 ? 5 * Math.sin(i / 9) * 0.3 : 0));
    blind.push(-96 + 20 * kb + g() * (3 + 8 * decay));
    threshold.push(-60 + g() * 1.2);
    fullBreak.push(Math.max(0, 1.9 - 1.5 * k + g() * 0.12 * (1 + decay)));
    blindBreak.push(Math.max(0, 2.1 - 0.15 * kb + g() * 0.18));
    thresholdBreak.push(Math.max(0, 0.05 + Math.abs(g()) * 0.03));
  }
  return { iters, full, blind, threshold, fullBreak, blindBreak, thresholdBreak };
}
