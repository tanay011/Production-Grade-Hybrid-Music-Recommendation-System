import os
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
from numpy import load
from scipy.sparse import load_npz

from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
APP_URL = os.getenv("APP_URL")


def _normalize_song_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["name"] = normalized["name"].astype(str).str.lower().str.strip()
    normalized["artist"] = normalized["artist"].astype(str).str.lower().str.strip()
    return normalized


@pytest.fixture(scope="module")
def recommender_artifacts():
    songs_data = _normalize_song_columns(pd.read_csv(DATA_DIR / "cleaned_data.csv"))
    filtered_data = _normalize_song_columns(
        pd.read_csv(DATA_DIR / "collab_filtered_data.csv")
    )

    return {
        "songs_data": songs_data,
        "filtered_data": filtered_data,
        "transformed_data": load_npz(DATA_DIR / "transformed_data.npz"),
        "transformed_hybrid_data": load_npz(DATA_DIR / "transformed_hybrid_data.npz"),
        "interaction_matrix": load_npz(DATA_DIR / "interaction_matrix.npz"),
        "track_ids": load(DATA_DIR / "track_ids.npy", allow_pickle=True),
    }


def _wait_for_app(url: str, timeout_seconds: int = 90) -> requests.Response:
    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response
            last_error = AssertionError(
                f"Expected status 200 from {url}, got {response.status_code}"
            )
        except requests.RequestException as exc:
            last_error = exc

        time.sleep(1)

    if last_error:
        raise AssertionError(f"App did not become ready at {url}") from last_error

    raise AssertionError(f"App did not become ready at {url}")


def test_recommendation_artifacts_are_aligned(recommender_artifacts):
    songs_data = recommender_artifacts["songs_data"]
    filtered_data = recommender_artifacts["filtered_data"]
    transformed_data = recommender_artifacts["transformed_data"]
    transformed_hybrid_data = recommender_artifacts["transformed_hybrid_data"]
    interaction_matrix = recommender_artifacts["interaction_matrix"]
    track_ids = recommender_artifacts["track_ids"]

    assert transformed_data.shape[0] == len(songs_data)
    assert transformed_hybrid_data.shape[0] == len(filtered_data)
    assert interaction_matrix.shape[0] == len(track_ids)
    assert len(filtered_data) == len(track_ids)


def test_content_recommendations_return_ranked_results(recommender_artifacts):
    songs_data = recommender_artifacts["songs_data"]
    transformed_data = recommender_artifacts["transformed_data"]
    sample_song = songs_data.iloc[0]

    recommendations = content_recommendation(
        song_name=sample_song["name"],
        artist_name=sample_song["artist"],
        songs_data=songs_data,
        transformed_data=transformed_data,
        k=5,
    )

    assert len(recommendations) >= 5
    assert {"name", "artist", "spotify_preview_url"}.issubset(recommendations.columns)
    assert recommendations[["name", "artist"]].notna().all().all()
    assert recommendations.drop_duplicates(subset=["name", "artist"]).shape[0] >= 5


def test_hybrid_recommendations_return_ranked_results(recommender_artifacts):
    filtered_data = recommender_artifacts["filtered_data"]
    transformed_hybrid_data = recommender_artifacts["transformed_hybrid_data"]
    interaction_matrix = recommender_artifacts["interaction_matrix"]
    track_ids = recommender_artifacts["track_ids"]
    sample_song = filtered_data.iloc[0]

    recommender = HybridRecommenderSystem(
        number_of_recommendations=5,
        weight_content_based=0.5,
    )
    recommendations = recommender.give_recommendations(
        song_name=sample_song["name"],
        artist_name=sample_song["artist"],
        songs_data=filtered_data,
        transformed_matrix=transformed_hybrid_data,
        track_ids=track_ids,
        interaction_matrix=interaction_matrix,
    )

    assert len(recommendations) >= 5
    assert {"name", "artist", "spotify_preview_url"}.issubset(recommendations.columns)
    assert recommendations[["name", "artist"]].notna().all().all()
    assert recommendations.drop_duplicates(subset=["name", "artist"]).shape[0] >= 5


@pytest.mark.skipif(not APP_URL, reason="Set APP_URL to run the live app smoke test.")
def test_running_app_responds():
    response = _wait_for_app(APP_URL)
    assert response.status_code == 200
