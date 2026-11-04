import {
  BreakdownInterval,
  EpisodeLog,
  MaintenanceInterval,
  OperationInterval,
  ParsedEpisode,
  RiskSample,
} from "@/types/eventLog";

export function parseEpisodeLog(log: EpisodeLog): ParsedEpisode {
  const { meta, events } = log;
  const operations: OperationInterval[] = [];
  const maintenances: MaintenanceInterval[] = [];
  const breakdowns: BreakdownInterval[] = [];
  const riskByMachine: Record<number, RiskSample[]> = {};

  for (let m = 0; m < meta.n_machines; m++) {
    riskByMachine[m] = [];
  }

  const activeOpByMachine = new Map<
    number,
    { job: number; op: number; start: number }
  >();
  const activeMaintByMachine = new Map<
    number,
    { start: number; riskBefore?: number }
  >();
  const pendingBreakdownByMachine = new Map<
    number,
    {
      job?: number;
      op?: number;
      breakdownTime: number;
      riskAtBreakdown?: number;
    }
  >();

  let maxTime = 0;

  for (const ev of events) {
    if (ev.t > maxTime) {
      maxTime = ev.t;
    }

    const m = ev.machine;

    if (ev.type === "risk_update" && m !== undefined && ev.risk !== undefined) {
      riskByMachine[m] ??= [];
      riskByMachine[m].push({ t: ev.t, risk: ev.risk });

      // If machine was in breakdown repair and risk drops/updates at repair end
      const pendingBd = pendingBreakdownByMachine.get(m);
      if (pendingBd && ev.t > pendingBd.breakdownTime) {
        breakdowns.push({
          id: `bd-m${m}-${pendingBd.breakdownTime}`,
          machine: m,
          job: pendingBd.job,
          op: pendingBd.op,
          breakdownTime: pendingBd.breakdownTime,
          repairEndTime: ev.t,
          riskAtBreakdown: pendingBd.riskAtBreakdown,
          riskAfterRepair: ev.risk,
        });
        pendingBreakdownByMachine.delete(m);
      }
    } else if (
      ev.type === "op_start" &&
      m !== undefined &&
      ev.job !== undefined &&
      ev.op !== undefined
    ) {
      // Close any unclosed breakdown if an op starts directly
      const pendingBd = pendingBreakdownByMachine.get(m);
      if (pendingBd && ev.t >= pendingBd.breakdownTime) {
        breakdowns.push({
          id: `bd-m${m}-${pendingBd.breakdownTime}`,
          machine: m,
          job: pendingBd.job,
          op: pendingBd.op,
          breakdownTime: pendingBd.breakdownTime,
          repairEndTime: ev.t,
          riskAtBreakdown: pendingBd.riskAtBreakdown,
        });
        pendingBreakdownByMachine.delete(m);
      }

      activeOpByMachine.set(m, { job: ev.job, op: ev.op, start: ev.t });
    } else if (ev.type === "op_end" && m !== undefined) {
      const active = activeOpByMachine.get(m);
      if (active) {
        operations.push({
          id: `op-m${m}-j${active.job}-o${active.op}-${active.start}`,
          machine: m,
          job: active.job,
          op: active.op,
          start: active.start,
          end: ev.t,
          interrupted: false,
        });
        activeOpByMachine.delete(m);
      }
    } else if (ev.type === "breakdown" && m !== undefined) {
      const active = activeOpByMachine.get(m);
      if (active) {
        operations.push({
          id: `op-interrupted-m${m}-j${active.job}-o${active.op}-${active.start}`,
          machine: m,
          job: active.job,
          op: active.op,
          start: active.start,
          end: ev.t,
          interrupted: true,
        });
        activeOpByMachine.delete(m);
      }
      pendingBreakdownByMachine.set(m, {
        job: ev.job ?? active?.job,
        op: ev.op ?? active?.op,
        breakdownTime: ev.t,
        riskAtBreakdown: ev.risk,
      });
    } else if (ev.type === "maintenance_start" && m !== undefined) {
      activeMaintByMachine.set(m, { start: ev.t, riskBefore: ev.risk });
    } else if (ev.type === "maintenance_end" && m !== undefined) {
      const activeMaint = activeMaintByMachine.get(m);
      if (activeMaint) {
        maintenances.push({
          id: `pm-m${m}-${activeMaint.start}`,
          machine: m,
          start: activeMaint.start,
          end: ev.t,
          riskBefore: activeMaint.riskBefore,
          riskAfter: ev.risk,
        });
        activeMaintByMachine.delete(m);
      }
    }
  }

  // Close any trailing breakdown at makespan
  for (const [m, pendingBd] of pendingBreakdownByMachine.entries()) {
    breakdowns.push({
      id: `bd-m${m}-${pendingBd.breakdownTime}`,
      machine: m,
      job: pendingBd.job,
      op: pendingBd.op,
      breakdownTime: pendingBd.breakdownTime,
      repairEndTime: maxTime,
      riskAtBreakdown: pendingBd.riskAtBreakdown,
    });
  }

  return {
    meta,
    makespan: maxTime,
    operations,
    maintenances,
    breakdowns,
    riskByMachine,
  };
}

/**
 * Evaluate the interpolated/stepwise risk of a machine at simulation time `t`.
 */
export function getMachineRiskAtTime(
  parsed: ParsedEpisode,
  machine: number,
  t: number
): number {
  const samples = parsed.riskByMachine[machine] ?? [];
  if (samples.length === 0) return 0;
  if (t <= samples[0].t) return samples[0].risk;

  let prev = samples[0];
  for (let i = 1; i < samples.length; i++) {
    const curr = samples[i];
    if (t < curr.t) {
      // If there's a maintenance reset or repair end at curr.t, hold previous risk until reset completes
      if (curr.risk < prev.risk - 0.25) {
        return prev.risk;
      }
      // Smooth linear interpolation between consecutive increasing risk updates
      const span = curr.t - prev.t;
      if (span <= 0) return curr.risk;
      const ratio = (t - prev.t) / span;
      return prev.risk + ratio * (curr.risk - prev.risk);
    }
    prev = curr;
  }
  return prev.risk;
}

export function getMachineStatusAtTime(
  parsed: ParsedEpisode,
  machine: number,
  t: number
): {
  state: "operating" | "maintenance" | "breakdown" | "idle";
  detail: string;
} {
  for (const bd of parsed.breakdowns) {
    if (bd.machine === machine && t >= bd.breakdownTime && t < bd.repairEndTime) {
      return {
        state: "breakdown",
        detail: `Repairing Breakdown (${bd.repairEndTime - bd.breakdownTime}u)`,
      };
    }
  }
  for (const pm of parsed.maintenances) {
    if (pm.machine === machine && t >= pm.start && t < pm.end) {
      return {
        state: "maintenance",
        detail: `Preventive Maint. (${pm.end - pm.start}u)`,
      };
    }
  }
  for (const op of parsed.operations) {
    if (op.machine === machine && t >= op.start && t < op.end) {
      return {
        state: "operating",
        detail: `Job ${op.job} · Op ${op.op}`,
      };
    }
  }
  return { state: "idle", detail: "Idle" };
}
