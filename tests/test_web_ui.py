import gradio as gr
from typer.testing import CliRunner

from src.utils.cli import app
from src.utils.config import Config
from src.utils.web_ui import launch_web_ui


def test_launch_web_ui_returns_interface(monkeypatch):
    config = Config()
    iface = launch_web_ui(config, mock=True, _launch=False)
    assert isinstance(iface, gr.Interface)


def test_web_command_invokes_launch(monkeypatch):
    called = {}

    def fake_launch(config, mock, _launch=True):
        called["mock"] = mock
        return None

    monkeypatch.setattr("src.utils.cli.launch_web_ui", fake_launch)
    runner = CliRunner()
    result = runner.invoke(app, ["web", "--mock"])
    assert result.exit_code == 0
    assert called.get("mock") is True
