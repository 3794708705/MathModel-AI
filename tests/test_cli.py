from pathlib import Path
from typing import Any

import pytest

from mathmodel_ai import cli
from mathmodel_ai.startup import SecretStoreStatus, StartupPreflightReport


def _report(tmp_path: Path) -> StartupPreflightReport:
    return StartupPreflightReport(
        python_version="3.12.13",
        python_executable=tmp_path / ".venv" / "Scripts" / "python.exe",
        database_url="sqlite+pysqlite:///:memory:",
        database_reachable=True,
        secret_store_status=SecretStoreStatus.NOT_CONFIGURED,
        stored_secret_count=0,
        live_provider_configured=False,
        directories=(tmp_path,),
        warnings=("Live provider not configured; the Web UI can still start.",),
    )


def test_cli_help_is_available_without_constructing_the_app(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])
    assert raised.value.code == 0
    assert "serve" in capsys.readouterr().out


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_cli_rejects_invalid_ports(port: str, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["serve", "--port", port])
    assert raised.value.code == 2
    assert "port" in capsys.readouterr().err


def test_serve_uses_the_canonical_factory_and_forwards_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(cli, "run_startup_preflight", lambda _settings: _report(tmp_path))
    monkeypatch.setattr(cli, "print_startup_preflight", lambda _report: None)
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    result = cli.serve(cli.ServeConfig(host="0.0.0.0", port=8123, reload=True))

    assert result == 0
    assert calls == [
        (
            "mathmodel_ai.main:create_app",
            {
                "factory": True,
                "host": "0.0.0.0",
                "port": 8123,
                "reload": True,
                "reload_dirs": [str(Path(cli.__file__).resolve().parents[1])],
            },
        )
    ]


def test_cli_main_builds_typed_serve_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[cli.ServeConfig] = []
    monkeypatch.setattr(cli, "serve", lambda config: received.append(config) or 0)

    assert cli.main(["serve", "--host", "127.0.0.2", "--port", "9000", "--reload"]) == 0
    assert received == [cli.ServeConfig(host="127.0.0.2", port=9000, reload=True)]
