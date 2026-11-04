export type Regime = "low" | "medium" | "high";

export type PolicyName =
  | "blind_ppo"
  | "threshold_spt"
  | "full_ppo"
  | "spt"
  | "lpt"
  | "fifo";

export type EventType =
  | "op_start"
  | "op_end"
  | "risk_update"
  | "maintenance_start"
  | "maintenance_end"
  | "breakdown"
  | "episode_end";

export interface EpisodeMeta {
  instance: string;
  regime: Regime;
  seed: number;
  policy: PolicyName;
  n_jobs: number;
  n_machines: number;
}

export interface ScheduleEvent {
  t: number;
  type: EventType;
  machine?: number;
  job?: number;
  op?: number;
  risk?: number;
}

export interface EpisodeLog {
  meta: EpisodeMeta;
  events: ScheduleEvent[];
}

export interface OperationInterval {
  id: string;
  machine: number;
  job: number;
  op: number;
  start: number;
  end: number;
  interrupted: boolean;
}

export interface MaintenanceInterval {
  id: string;
  machine: number;
  start: number;
  end: number;
  riskBefore?: number;
  riskAfter?: number;
}

export interface BreakdownInterval {
  id: string;
  machine: number;
  job?: number;
  op?: number;
  breakdownTime: number;
  repairEndTime: number;
  riskAtBreakdown?: number;
  riskAfterRepair?: number;
}

export interface RiskSample {
  t: number;
  risk: number;
}

export interface ParsedEpisode {
  meta: EpisodeMeta;
  makespan: number;
  operations: OperationInterval[];
  maintenances: MaintenanceInterval[];
  breakdowns: BreakdownInterval[];
  riskByMachine: Record<number, RiskSample[]>;
}
