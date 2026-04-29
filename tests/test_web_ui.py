"""Tests for the browser UI route."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_web_ui_serves_index() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Internal Knowledge Chat" in response.text


def test_static_asset_is_served() -> None:
    response = TestClient(app).get("/static/app.js")

    assert response.status_code == 200
    assert "refreshHealth" in response.text
