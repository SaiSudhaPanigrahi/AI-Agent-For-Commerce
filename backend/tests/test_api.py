from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


SAMPLE_CATALOG = [
    {
        "id": "item-1",
        "title": "Brown Everyday Tote",
        "category": "bags",
        "color": "brown",
        "price": 59.0,
        "description": "Daily tote bag",
        "image_path": "bags/bag1_brown.jpg",
        "tags": ["tote", "commute"],
    },
    {
        "id": "item-2",
        "title": "Pink Casual Sneakers",
        "category": "shoes",
        "color": "pink",
        "price": 79.0,
        "description": "Casual sneakers",
        "image_path": "shoes/shoe4_pink.jpg",
        "tags": ["casual", "everyday"],
    },
]


def _install_test_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(SAMPLE_CATALOG))

    catalog_loader_mod = types.ModuleType("services.catalog_loader")

    def ensure_catalog(data_dir: Path, cache_dir: Path, regenerate: bool = False) -> Path:
        return catalog_file

    catalog_loader_mod.ensure_catalog = ensure_catalog

    text_index_mod = types.ModuleType("services.text_index")

    class TextIndex:
        def __init__(self, catalog_path: Path, cache_dir: Path, force_rebuild: bool = False):
            self.catalog = json.loads(Path(catalog_path).read_text())

        def search_with_filters(
            self,
            query: str,
            category: str | None = None,
            color: str | None = None,
            min_price: float | None = None,
            max_price: float | None = None,
            top_k: int = 12,
        ):
            items = []
            for item in self.catalog:
                if category and item["category"] != category:
                    continue
                if color and item["color"] != color:
                    continue
                if min_price is not None and item["price"] < float(min_price):
                    continue
                if max_price is not None and item["price"] > float(max_price):
                    continue
                item_with_score = dict(item)
                item_with_score["score"] = 0.95
                items.append(item_with_score)
            return items[:top_k]

    text_index_mod.TextIndex = TextIndex

    vision_search_mod = types.ModuleType("services.vision_search")

    class VisionIndex:
        def __init__(self, catalog_path: Path, cache_dir: Path, data_dir: Path, force_rebuild: bool = False):
            self.catalog = json.loads(Path(catalog_path).read_text())

        def search_image_path(self, path: Path, top_k: int = 8):
            return self.catalog[:top_k]

        def search_image_url(self, url: str, top_k: int = 8):
            if "invalid" in url:
                return []
            return self.catalog[:top_k]

    vision_search_mod.VisionIndex = VisionIndex

    agent_mod = types.ModuleType("agent.agent")

    class Agent:
        def __init__(self, text_index: TextIndex, vision_index: VisionIndex):
            self.text_index = text_index

        async def chat(self, message: str):
            results = self.text_index.search_with_filters(message, top_k=2)
            return {
                "intent": "recommend",
                "source": "test",
                "text": "Mocked chat response",
                "reply": "Mocked chat response",
                "results": results,
                "filters": {"category": None, "color": None},
            }

    agent_mod.Agent = Agent

    path_repair_mod = types.ModuleType("services.path_repair")

    def repair_paths(catalog_path: Path, data_dir: Path) -> int:
        return 2

    path_repair_mod.repair_paths = repair_paths

    monkeypatch.setitem(sys.modules, "services.catalog_loader", catalog_loader_mod)
    monkeypatch.setitem(sys.modules, "services.text_index", text_index_mod)
    monkeypatch.setitem(sys.modules, "services.vision_search", vision_search_mod)
    monkeypatch.setitem(sys.modules, "agent.agent", agent_mod)
    monkeypatch.setitem(sys.modules, "services.path_repair", path_repair_mod)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    _install_test_fakes(monkeypatch, tmp_path)
    sys.modules.pop("app", None)
    app_mod = importlib.import_module("app")
    return TestClient(app_mod.app)


def test_catalog_returns_items(client: TestClient):
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["image_path"] == "bags/bag1_brown.jpg"


def test_search_text_accepts_alias_query_key(client: TestClient):
    resp = client.post("/api/search_text", json={"query": "bags", "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["q"] == "bags"
    assert body["k"] == 5
    assert len(body["results"]) >= 1


def test_search_text_rejects_invalid_price_range(client: TestClient):
    resp = client.post("/api/search_text", json={"q": "bags", "min_price": 100, "max_price": 10})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "Request failed"
    assert "min_price" in body["detail"]


def test_search_text_validates_k_upper_bound(client: TestClient):
    resp = client.post("/api/search_text", json={"q": "bags", "k": 999})
    assert resp.status_code == 422
    assert resp.json()["error"] == "Validation error"


def test_search_by_url_validates_url_shape(client: TestClient):
    resp = client.post("/api/search_by_url", json={"url": "not-a-url", "k": 5})
    assert resp.status_code == 422
    assert resp.json()["error"] == "Validation error"


def test_search_by_url_returns_results(client: TestClient):
    resp = client.post("/api/search_by_url", json={"url": "https://example.com/pink-shoe.jpg", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2


def test_chat_rejects_blank_message(client: TestClient):
    resp = client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "message must not be blank"


def test_reindex_returns_success_response(client: TestClient):
    resp = client.post("/api/reindex")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "Regenerated catalog" in body["message"]
