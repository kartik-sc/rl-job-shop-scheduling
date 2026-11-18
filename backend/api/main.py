from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.mock_data import get_mock_episode

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


@app.get("/episodes")
@app.get("/api/episodes")
def list_episodes() -> list[dict[str, Any]]:
    mock = get_mock_episode()
    return [
        {
            "id": "mock",
            "label": "Mock Episode (6M × 5J, 1 PM, 1 Breakdown)",
            "meta": mock["meta"],
        }
    ]


@app.get("/episodes/mock")
@app.get("/api/episodes/mock")
def read_mock_episode() -> dict[str, Any]:
    return get_mock_episode()


@app.get("/episodes/{episode_id}")
@app.get("/api/episodes/{episode_id}")
def read_episode(episode_id: str) -> dict[str, Any]:
    if episode_id == "mock":
        return get_mock_episode()
    raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found")
