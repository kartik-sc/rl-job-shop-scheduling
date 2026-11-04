import { Scene, SceneJob, SceneMachine } from "@/lib/scene";
import { mulberry32 } from "@/lib/viz";

export type SimPolicy = "manual" | "threshold" | "riskaware";

interface SimMachine {
  id: number;
  state: "idle" | "operating" | "maintenance" | "breakdown";
  risk: number;
  remaining: number;
  total: number;
  job: number | null;
}

interface SimJob {
  id: number;
  route: { m: number; d: number }[];
  k: number;
  status: "queued" | "processing" | "transit" | "blocked" | "done";
  machine: number | null;
  from?: number;
  to?: number;
  tStart?: number;
  tEnd?: number;
  queuedAt: number;
}

export interface Sim {
  t: number;
  n: number;
  machines: SimMachine[];
  jobs: SimJob[];
  breakdowns: number;
  pms: number;
  completedOps: number;
  reward: number;
  lastReward: number;
  finishedAt: number | null;
  nextJobId: number;
  rnd: () => number;
  log: string[];
}

const TRANSIT = 2.5;
const PM_TIME = 8;
const REPAIR_TIME = 40;

function makeRoute(rnd: () => number, n: number, len = 4) {
  const ms = Array.from({ length: n }, (_, i) => i).sort(() => rnd() - 0.5).slice(0, len);
  return ms.map((m) => ({ m, d: 6 + Math.round(rnd() * 10) }));
}

export function newSim(n = 6, nJobs = 6, seed = 11): Sim {
  const rnd = mulberry32(seed);
  const machines: SimMachine[] = Array.from({ length: n }, (_, id) => ({
    id,
    state: "idle",
    risk: 0.05 + rnd() * 0.25,
    remaining: 0,
    total: 1,
    job: null,
  }));
  const jobs: SimJob[] = Array.from({ length: nJobs }, (_, id) => {
    const route = makeRoute(rnd, n);
    return { id, route, k: 0, status: "queued", machine: route[0].m, queuedAt: id * 0.01 };
  });
  return {
    t: 0,
    n,
    machines,
    jobs,
    breakdowns: 0,
    pms: 0,
    completedOps: 0,
    reward: 0,
    lastReward: 0,
    finishedAt: null,
    nextJobId: nJobs,
    rnd,
    log: ["episode started"],
  };
}

export function addRushJob(s: Sim) {
  const route = makeRoute(s.rnd, s.n);
  s.jobs.push({ id: s.nextJobId++, route, k: 0, status: "queued", machine: route[0].m, queuedAt: s.t });
  s.finishedAt = null;
  s.log.unshift(`rush job J${s.nextJobId - 1} injected`);
}

export function forceBreakdown(s: Sim, m: number) {
  const mc = s.machines[m];
  if (mc.state === "breakdown") return;
  breakMachine(s, mc);
}

export function forceMaintenance(s: Sim, m: number) {
  const mc = s.machines[m];
  if (mc.state !== "idle") return;
  startPM(s, mc);
}

function breakMachine(s: Sim, mc: SimMachine) {
  if (mc.job !== null) {
    const j = s.jobs.find((x) => x.id === mc.job);
    if (j) {
      j.status = "blocked";
      j.machine = mc.id;
    }
  }
  mc.state = "breakdown";
  mc.remaining = REPAIR_TIME;
  mc.total = REPAIR_TIME;
  s.breakdowns++;
  s.reward -= 6;
  s.lastReward = -6;
  s.log.unshift(`t=${s.t.toFixed(0)} M${mc.id} FAILED (risk ${(mc.risk * 100).toFixed(0)}%)`);
}

function startPM(s: Sim, mc: SimMachine) {
  mc.state = "maintenance";
  mc.remaining = PM_TIME;
  mc.total = PM_TIME;
  mc.job = null;
  s.pms++;
  s.reward -= 0.8;
  s.lastReward = -0.8;
  s.log.unshift(`t=${s.t.toFixed(0)} M${mc.id} → preventive maintenance`);
}

export function step(s: Sim, dt: number, policy: SimPolicy, stress: number) {
  s.t += dt;
  s.reward -= 0.04 * dt;

  for (const j of s.jobs) {
    if (j.status === "transit" && j.tEnd !== undefined && s.t >= j.tEnd) {
      j.status = "queued";
      j.machine = j.to ?? null;
      j.queuedAt = s.t;
    }
  }

  for (const mc of s.machines) {
    if (mc.state === "operating") {
      mc.remaining -= dt;
      mc.risk = Math.min(1, mc.risk + 0.011 * stress * dt);
      const hazard = 0.0012 * stress * Math.exp(4.2 * mc.risk);
      if (s.rnd() < 1 - Math.exp(-hazard * dt)) {
        breakMachine(s, mc);
        continue;
      }
      if (mc.remaining <= 0) {
        const j = s.jobs.find((x) => x.id === mc.job)!;
        s.completedOps++;
        s.reward += 1;
        s.lastReward = 1;
        j.k++;
        mc.state = "idle";
        mc.job = null;
        if (j.k >= j.route.length) {
          j.status = "done";
          j.machine = null;
        } else {
          const next = j.route[j.k].m;
          if (next === mc.id) {
            j.status = "queued";
            j.machine = next;
            j.queuedAt = s.t;
          } else {
            j.status = "transit";
            j.from = mc.id;
            j.to = next;
            j.tStart = s.t;
            j.tEnd = s.t + TRANSIT;
            j.machine = null;
          }
        }
      }
    } else if (mc.state === "maintenance") {
      mc.remaining -= dt;
      mc.risk = Math.max(0.04, mc.risk - (0.9 * dt) / PM_TIME);
      if (mc.remaining <= 0) {
        mc.state = "idle";
        mc.risk = 0.05;
        s.log.unshift(`t=${s.t.toFixed(0)} M${mc.id} back online (risk reset)`);
      }
    } else if (mc.state === "breakdown") {
      mc.remaining -= dt;
      if (mc.remaining <= 0) {
        mc.state = "idle";
        mc.risk = 0.08;
        for (const j of s.jobs) {
          if (j.status === "blocked" && j.machine === mc.id) {
            j.status = "queued";
            j.queuedAt = -1;
          }
        }
        s.log.unshift(`t=${s.t.toFixed(0)} M${mc.id} repaired`);
      }
    }
  }

  for (const mc of s.machines) {
    if (mc.state !== "idle") continue;
    const queue = s.jobs.filter((j) => j.status === "queued" && j.machine === mc.id);

    if (policy === "threshold" && mc.risk >= 0.7) {
      startPM(s, mc);
      continue;
    }
    if (policy === "riskaware") {
      const urgent = mc.risk >= 0.62;
      const early = mc.risk >= 0.45 && queue.length === 0;
      if (urgent || early) {
        startPM(s, mc);
        continue;
      }
    }
    if (queue.length === 0) continue;
    queue.sort((a, b) => a.route[a.k].d - b.route[b.k].d || a.queuedAt - b.queuedAt);
    const j = queue[0];
    const d = j.route[j.k].d;
    mc.state = "operating";
    mc.job = j.id;
    mc.remaining = d;
    mc.total = d;
    j.status = "processing";
  }

  const allDone = s.jobs.every((j) => j.status === "done");
  if (allDone && s.finishedAt === null) {
    s.finishedAt = s.t;
    s.log.unshift(`t=${s.t.toFixed(0)} all jobs complete · makespan ${s.t.toFixed(0)}`);
  }
  if (s.log.length > 30) s.log.length = 30;
}

export function simScene(s: Sim): Scene {
  const machines: SceneMachine[] = s.machines.map((m) => ({
    id: m.id,
    state: m.state,
    risk: m.risk,
    progress: m.state === "idle" ? 0 : 1 - m.remaining / m.total,
    jobId: m.state === "operating" ? m.job : null,
    detail:
      m.state === "operating"
        ? `Running job ${m.job}`
        : m.state === "maintenance"
          ? "Preventive maintenance"
          : m.state === "breakdown"
            ? `Repairing (${Math.ceil(m.remaining)}u left)`
            : "Idle",
  }));
  const counters = new Map<number, number>();
  const flows = new Set<string>();
  const flowList: [number, number][] = [];
  const jobs: SceneJob[] = s.jobs.map((j) => {
    for (let i = 0; i < j.route.length - 1; i++) {
      const key = `${j.route[i].m}-${j.route[i + 1].m}`;
      if (!flows.has(key)) {
        flows.add(key);
        flowList.push([j.route[i].m, j.route[i + 1].m]);
      }
    }
    let queueIndex = 0;
    if (j.status === "queued" && j.machine !== null) {
      queueIndex = counters.get(j.machine) ?? 0;
      counters.set(j.machine, queueIndex + 1);
    }
    return {
      id: j.id,
      status: j.status,
      machine: j.machine,
      from: j.from,
      to: j.to,
      p:
        j.status === "transit" && j.tStart !== undefined && j.tEnd !== undefined
          ? (s.t - j.tStart) / (j.tEnd - j.tStart)
          : undefined,
      opIdx: j.k,
      nOps: j.route.length,
      queueIndex,
    };
  });
  return {
    t: s.t,
    machines,
    jobs,
    flows: flowList,
    completedOps: s.completedOps,
    totalOps: s.jobs.reduce((a, j) => a + j.route.length, 0),
    doneJobs: s.jobs.filter((j) => j.status === "done").length,
  };
}
