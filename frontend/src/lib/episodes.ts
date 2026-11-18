import { EpisodeLog, EpisodeSummary } from "@/types/eventLog";
import {
  BLIND_PPO_EPISODE_LOG,
  MOCK_EPISODE_LOG,
  THRESHOLD_SPT_EPISODE_LOG,
} from "@/lib/mockEpisode";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const MOCK_LABEL = "MOCK DATA";

const LOCAL_MOCKS: Record<string, { log: EpisodeLog; name: string }> = {
  "local-mock-full_ppo": { log: MOCK_EPISODE_LOG, name: "Full PPO" },
  "local-mock-threshold_spt": {
    log: THRESHOLD_SPT_EPISODE_LOG,
    name: "Threshold+SPT",
  },
  "local-mock-blind_ppo": { log: BLIND_PPO_EPISODE_LOG, name: "Blind PPO" },
};

function summariseLocal(id: string): EpisodeSummary {
  const { log, name } = LOCAL_MOCKS[id];
  const end = log.events.find((e) => e.type === "episode_end");
  const count = (type: string) =>
    log.events.filter((e) => e.type === type).length;
  return {
    id,
    source: "mock",
    label: `${MOCK_LABEL}: ${name} (hand-written)`,
    detail: `${log.meta.instance} · bundled with the frontend, not a simulation result`,
    meta: log.meta,
    makespan: end?.t ?? 0,
    maintenances: count("maintenance_start"),
    breakdowns: count("breakdown"),
    maintenance_downtime: null,
    repair_downtime: null,
  };
}

/** Mocks the API does not serve, shown even when it is up. */
const EXTRA_LOCAL_MOCKS = ["local-mock-threshold_spt", "local-mock-blind_ppo"];

export function localEpisodes(apiAvailable: boolean): EpisodeSummary[] {
  const ids = apiAvailable
    ? EXTRA_LOCAL_MOCKS
    : ["local-mock-full_ppo", ...EXTRA_LOCAL_MOCKS];
  return ids.map(summariseLocal);
}

export function isLocal(id: string): boolean {
  return id in LOCAL_MOCKS;
}

export async function fetchEpisodes(): Promise<EpisodeSummary[]> {
  const res = await fetch(`${API_BASE_URL}/episodes`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /episodes: HTTP ${res.status}`);
  return (await res.json()) as EpisodeSummary[];
}

export async function loadEpisode(id: string): Promise<EpisodeLog> {
  if (isLocal(id)) return LOCAL_MOCKS[id].log;
  const res = await fetch(
    `${API_BASE_URL}/episodes/${encodeURIComponent(id)}`,
    { cache: "no-store" }
  );
  if (!res.ok) throw new Error(`GET /episodes/${id}: HTTP ${res.status}`);
  return (await res.json()) as EpisodeLog;
}
