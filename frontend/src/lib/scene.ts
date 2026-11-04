import { ParsedEpisode } from "@/types/eventLog";
import { getMachineRiskAtTime, getMachineStatusAtTime } from "@/lib/parseEpisode";

export type MachineState = "idle" | "operating" | "maintenance" | "breakdown";
export type JobStatus =
  | "pending"
  | "queued"
  | "processing"
  | "transit"
  | "blocked"
  | "done";

export interface SceneMachine {
  id: number;
  state: MachineState;
  risk: number;
  progress: number; // 0..1 of the current operation / repair / PM
  jobId: number | null;
  detail: string;
}

export interface SceneJob {
  id: number;
  status: JobStatus;
  machine: number | null; // machine for queued/processing/blocked
  from?: number;
  to?: number;
  p?: number; // transit progress 0..1
  opIdx: number; // ops finished
  nOps: number;
  queueIndex: number;
}

export interface Scene {
  t: number;
  machines: SceneMachine[];
  jobs: SceneJob[];
  flows: [number, number][];
  completedOps: number;
  totalOps: number;
  doneJobs: number;
}

const TRANSIT = 2.5;

interface RouteStep {
  machine: number;
  start: number;
  end: number;
  op: number;
}

export function buildRoutes(parsed: ParsedEpisode): Map<number, RouteStep[]> {
  const routes = new Map<number, RouteStep[]>();
  for (const op of parsed.operations) {
    if (op.interrupted) continue;
    const list = routes.get(op.job) ?? [];
    list.push({ machine: op.machine, start: op.start, end: op.end, op: op.op });
    routes.set(op.job, list);
  }
  for (const list of routes.values()) list.sort((a, b) => a.op - b.op);
  return routes;
}

export function buildFlows(routes: Map<number, RouteStep[]>): [number, number][] {
  const seen = new Set<string>();
  const out: [number, number][] = [];
  for (const r of routes.values()) {
    for (let i = 0; i < r.length - 1; i++) {
      const k = `${r[i].machine}-${r[i + 1].machine}`;
      if (r[i].machine !== r[i + 1].machine && !seen.has(k)) {
        seen.add(k);
        out.push([r[i].machine, r[i + 1].machine]);
      }
    }
  }
  return out;
}

export function buildEpisodeScene(
  parsed: ParsedEpisode,
  routes: Map<number, RouteStep[]>,
  flows: [number, number][],
  t: number
): Scene {
  const n = parsed.meta.n_machines;
  const machines: SceneMachine[] = [];
  for (let m = 0; m < n; m++) {
    const st = getMachineStatusAtTime(parsed, m, t);
    let progress = 0;
    let jobId: number | null = null;
    if (st.state === "operating") {
      const op = parsed.operations.find(
        (o) => o.machine === m && t >= o.start && t < o.end
      );
      if (op) {
        progress = (t - op.start) / Math.max(1e-6, op.end - op.start);
        jobId = op.job;
      }
    } else if (st.state === "maintenance") {
      const pm = parsed.maintenances.find(
        (p) => p.machine === m && t >= p.start && t < p.end
      );
      if (pm) progress = (t - pm.start) / (pm.end - pm.start);
    } else if (st.state === "breakdown") {
      const bd = parsed.breakdowns.find(
        (b) => b.machine === m && t >= b.breakdownTime && t < b.repairEndTime
      );
      if (bd) progress = (t - bd.breakdownTime) / (bd.repairEndTime - bd.breakdownTime);
    }
    machines.push({
      id: m,
      state: st.state,
      risk: getMachineRiskAtTime(parsed, m, t),
      progress,
      jobId,
      detail: st.detail,
    });
  }

  const jobs: SceneJob[] = [];
  let doneJobs = 0;
  let completedOps = 0;
  for (const [id, route] of routes.entries()) {
    const nOps = route.length;
    const finished = route.filter((s) => s.end <= t).length;
    completedOps += finished;
    const base: SceneJob = {
      id,
      status: "queued",
      machine: route[0].machine,
      opIdx: finished,
      nOps,
      queueIndex: 0,
    };
    const interrupted = parsed.operations.find(
      (o) => o.interrupted && o.job === id && t >= o.start
    );
    const restart = interrupted
      ? route.find((s) => s.op === interrupted.op)
      : undefined;
    if (interrupted && restart && t >= interrupted.start && t < restart.start && t >= interrupted.end) {
      jobs.push({ ...base, status: "blocked", machine: interrupted.machine });
      continue;
    }
    const proc = route.find((s) => t >= s.start && t < s.end);
    const procInt = parsed.operations.find(
      (o) => o.interrupted && o.job === id && t >= o.start && t < o.end
    );
    if (proc || procInt) {
      jobs.push({ ...base, status: "processing", machine: (proc ?? procInt)!.machine });
      continue;
    }
    if (finished === nOps) {
      doneJobs++;
      jobs.push({ ...base, status: "done", machine: null });
      continue;
    }
    if (finished === 0) {
      jobs.push({ ...base, status: "queued", machine: route[0].machine });
      continue;
    }
    const prev = route[finished - 1];
    const next = route[finished];
    const tr = Math.min(TRANSIT, Math.max(0.5, next.start - prev.end));
    if (prev.machine !== next.machine && t < prev.end + tr) {
      jobs.push({
        ...base,
        status: "transit",
        machine: null,
        from: prev.machine,
        to: next.machine,
        p: (t - prev.end) / tr,
      });
    } else {
      jobs.push({ ...base, status: "queued", machine: next.machine });
    }
  }

  const counters = new Map<number, number>();
  jobs.sort((a, b) => a.id - b.id);
  for (const j of jobs) {
    if (j.status === "queued" && j.machine !== null) {
      const c = counters.get(j.machine) ?? 0;
      j.queueIndex = c;
      counters.set(j.machine, c + 1);
    }
  }
  const total = [...routes.values()].reduce((a, r) => a + r.length, 0);
  return { t, machines, jobs, flows, completedOps, totalOps: total, doneJobs };
}
