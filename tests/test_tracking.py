"""Tracking identity and lifecycle tests without network or model inference."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from non_verifiable_rewards.training.config import load_config, validate_config
from non_verifiable_rewards.training.tracking import tracking_run


@pytest.fixture
def sdk(monkeypatch):
    module = SimpleNamespace(run=None, Settings=lambda **kwargs: kwargs)

    def initialize(**kwargs):
        run = SimpleNamespace(
            entity=kwargs["entity"] or "test-account",
            url=f"https://wandb.ai/test-account/project/runs/{kwargs['id']}",
            finish=Mock(side_effect=lambda **kw: setattr(module, "run", None)),
        )
        module.run = run
        return run

    module.init = Mock(side_effect=initialize)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def enabled_config(tmp_path):
    return validate_config(
        {
            "output_dir": str(tmp_path / "sweep"),
            "wandb": {"enabled": True},
        }
    )


def test_standard_configs_enable_wandb_with_the_requested_project():
    for name in ("grpo_qwen3_4b", "eidf_grpo_qwen3_4b", "eidf_grpo_vllm_qwen3_4b"):
        options = load_config(f"configs/{name}.toml")["wandb"]
        assert options == dict(
            enabled=True, project="non-verifiable rewards", entity="", mode="online"
        )
    assert not load_config("configs/eidf_grpo_rouge_20261120.toml")["wandb"]["enabled"]


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "yes"},
        {"project": ""},
        {"project": "bad/name"},
        {"project": " padded "},
        {"entity": None},
        {"mode": "unknown"},
        {"api_key": "secret"},
    ],
)
def test_bad_tracking_configuration_fails_before_loading_models(options):
    with pytest.raises(ValueError):
        validate_config({"wandb": options})


def test_disabled_tracking_does_not_import_or_initialize_wandb(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    config = validate_config({})
    with tracking_run(config, config["runs"][0], tmp_path / "run") as tracked:
        assert tracked is None
    assert not (tmp_path / "run").exists()


def test_resume_reuses_identity_and_new_sweep_gets_new_identity(tmp_path, sdk):
    config = enabled_config(tmp_path)
    run = config["runs"][0]
    target = tmp_path / "run"
    with tracking_run(config, run, target) as first:
        identity = json.loads((target / "wandb.json").read_text())
        assert identity["entity"] == "test-account"
        assert identity["url"] == first.url
        assert sdk.init.call_args.kwargs["config"]["reward_run"] == run
        assert sdk.init.call_args.kwargs["resume"] == "allow"
    first.finish.assert_called_once_with(exit_code=0)
    with tracking_run(config, run, target):
        assert sdk.init.call_args.kwargs["id"] == identity["id"]
        assert sdk.init.call_args.kwargs["entity"] == "test-account"
    with tracking_run(config, run, tmp_path / "other-sweep"):
        assert sdk.init.call_args.kwargs["id"] != identity["id"]


def test_failed_training_finishes_run_and_restores_environment(
    tmp_path, sdk, monkeypatch
):
    config = enabled_config(tmp_path)
    monkeypatch.setenv("WANDB_LOG_MODEL", "checkpoint")
    monkeypatch.delenv("WANDB_WATCH", raising=False)
    with pytest.raises(RuntimeError, match="interrupted"):
        with tracking_run(config, config["runs"][0], tmp_path / "run") as tracked:
            assert os.environ["WANDB_LOG_MODEL"] == "false"
            assert os.environ["WANDB_WATCH"] == "false"
            raise RuntimeError("interrupted")
    tracked.finish.assert_called_once_with(exit_code=1)
    assert os.environ["WANDB_LOG_MODEL"] == "checkpoint"
    assert "WANDB_WATCH" not in os.environ


def test_failed_connection_retains_id_for_retry(tmp_path, sdk):
    config = enabled_config(tmp_path)
    sdk.init.side_effect = RuntimeError("connection failed")
    target = tmp_path / "run"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="connection failed"):
            with tracking_run(config, config["runs"][0], target):
                pytest.fail("Must not train without requested tracking")
    ids = [call.kwargs["id"] for call in sdk.init.call_args_list]
    assert ids[0] == ids[1] == json.loads((target / "wandb.json").read_text())["id"]


def test_active_run_is_not_reused_for_an_unrelated_reward(tmp_path, sdk):
    config = enabled_config(tmp_path)
    sdk.run = object()
    with pytest.raises(RuntimeError, match="active W&B run"):
        with tracking_run(config, config["runs"][0], tmp_path / "run"):
            pytest.fail("Must not mix different training runs")
    sdk.init.assert_not_called()


def test_offline_tracking_does_not_claim_server_resume(tmp_path, sdk):
    config = enabled_config(tmp_path)
    config["wandb"]["mode"] = "offline"
    with tracking_run(config, config["runs"][0], tmp_path / "run"):
        assert sdk.init.call_args.kwargs["mode"] == "offline"
        assert sdk.init.call_args.kwargs["resume"] is None


def test_tracking_destination_cannot_change_on_resume(tmp_path, sdk):
    config = enabled_config(tmp_path)
    target = tmp_path / "run"
    with tracking_run(config, config["runs"][0], target):
        pass
    config["wandb"]["project"] = "different-project"
    with pytest.raises(ValueError, match="project/entity changed"):
        with tracking_run(config, config["runs"][0], target):
            pytest.fail("Must not silently move a resumed run")
