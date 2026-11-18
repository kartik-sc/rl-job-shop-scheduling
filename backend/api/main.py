from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.episodes import EpisodeLog, EpisodeSummary, get_episode, list_episodes

app = FastAPI(title="RAJS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/episodes", response_model=list[EpisodeSummary])
@app.get("/api/episodes", response_model=list[EpisodeSummary])
def read_episodes() -> list[EpisodeSummary]:
    """Real simulator logs first, then the mock, whose label starts with MOCK DATA."""
    return list_episodes()


@app.get("/episodes/{episode_id}", response_model=EpisodeLog, response_model_exclude_none=True)
@app.get("/api/episodes/{episode_id}", response_model=EpisodeLog, response_model_exclude_none=True)
def read_episode(episode_id: str) -> EpisodeLog:
    log = get_episode(episode_id)
    if log is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found")
    return EpisodeLog.model_validate(log)
