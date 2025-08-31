from fastapi.testclient import TestClient

from src.api.server import app, config, plugin_manager
from src.plugins import initialize_plugins

client = TestClient(app)


def setup_function(function):
    initialize_plugins(config)


def test_list_plugins_endpoint():
    response = client.get("/api/plugins")
    assert response.status_code == 200
    data = response.json()
    assert any(p["name"] == "time_of_day" for p in data)


def test_set_plugin_state_endpoint():
    response = client.post("/api/plugins/time_of_day", json={"enabled": False})
    assert response.status_code == 200
    assert not plugin_manager.plugins["time_of_day"].enabled
    response = client.post("/api/plugins/time_of_day", json={"enabled": True})
    assert plugin_manager.plugins["time_of_day"].enabled
