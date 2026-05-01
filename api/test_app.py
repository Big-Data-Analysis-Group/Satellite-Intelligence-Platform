"""pytest test suite for the Satellite Intelligence Platform API."""

import pytest
from fastapi.testclient import TestClient
from api.app import app

VALID_SCENE = {
    "ndvi_mean": 0.30,
    "ndvi_std":  0.17,
    "ndbi_mean": 0.19,
    "ndbi_std":  0.13,
    "bsi_mean":  0.20,
    "bsi_std":   0.12,
}


@pytest.fixture(scope="module")
def client():
    # context-manager form triggers the lifespan (startup/shutdown) events
    with TestClient(app) as c:
        yield c


# ── happy-path tests ──────────────────────────────────────────────────────────

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "features" in data
    assert len(data["features"]) == 6
    assert "metrics" in data


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_SCENE)
    assert r.status_code == 200
    data = r.json()
    assert data["damage_label"] in (0, 1)
    assert 0.0 <= data["damage_probability"] <= 1.0
    assert "prediction_id" in data
    assert "model_version" in data


def test_predict_response_fields(client):
    r = client.post("/predict", json=VALID_SCENE)
    data = r.json()
    assert data["damage_label_text"] in ("damaged", "undamaged")
    assert data["model_version"] == "1.0.0"
    assert len(data["prediction_id"]) == 36  # UUID format


def test_batch_predict(client):
    payload = {"records": [VALID_SCENE] * 5}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 5
    assert len(data["predictions"]) == 5
    assert "processing_time_ms" in data


# ── validation / error tests ──────────────────────────────────────────────────

def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID_SCENE.items() if k != "ndvi_mean"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_wrong_type(client):
    bad = {**VALID_SCENE, "ndvi_mean": "not-a-number"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    bad = {**VALID_SCENE, "ndvi_mean": 5.0}  # max is 1.0
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_negative_std(client):
    bad = {**VALID_SCENE, "ndvi_std": -0.5}  # std must be >= 0
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_exceeds_limit(client):
    payload = {"records": [VALID_SCENE] * 101}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 422