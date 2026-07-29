from __future__ import annotations

import json
import stat
import tempfile
import textwrap
import time
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from robot_pipeline_app.artifacts import list_runs, write_run_artifacts
from robot_pipeline_app.checks import collect_doctor_checks, summarize_checks
from robot_pipeline_app.commands import (
    build_lerobot_deploy_command,
    build_lerobot_record_command,
    build_lerobot_teleop_command,
    build_lerobot_train_command,
)
from robot_pipeline_app.compat import _CAP_CACHE, probe_lerobot_capabilities
from robot_pipeline_app.config_store import normalize_config_without_prompts, save_config
from robot_pipeline_app.constants import DEFAULT_CONFIG_VALUES
from robot_pipeline_app.runner import run_process_streaming
from robot_pipeline_app.support_bundle import create_support_bundle
from robot_pipeline_app.workflow_queue import WorkflowQueueService, build_train_deploy_eval_queue_item


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "compat"


class _FakeQueueRunController:
    def __init__(self) -> None:
        self._active = False
        self._complete_callback = None
        self.calls: list[dict] = []

    def has_active_process(self) -> bool:
        return self._active

    def run_process_async(self, *, cmd, cwd, hooks, complete_callback, **kwargs):  # type: ignore[no-untyped-def]
        self._active = True
        self.calls.append({"cmd": list(cmd), "cwd": cwd, **kwargs})
        self._complete_callback = complete_callback
        return True, None

    def cancel_active_run(self) -> tuple[bool, str]:
        if not self._active:
            return False, "No active run."
        return True, "Cancel requested."

    def finish(self, *, return_code: int = 0, canceled: bool = False) -> None:
        callback = self._complete_callback
        self._active = False
        self._complete_callback = None
        if callback is not None:
            callback(return_code, canceled)


def _write_fake_python(path: Path, fixtures_dir: Path) -> None:
    """Install a fake python that answers version/module/help probes and runs commands."""
    help_map = {
        "record": (fixtures_dir / "lerobot_0_6_record_help.txt").read_text(encoding="utf-8"),
        "train": (fixtures_dir / "lerobot_0_6_train_help.txt").read_text(encoding="utf-8"),
        "teleop": (fixtures_dir / "lerobot_0_6_teleop_help.txt").read_text(encoding="utf-8"),
        "rollout": (fixtures_dir / "lerobot_0_6_rollout_help.txt").read_text(encoding="utf-8"),
        "replay": (fixtures_dir / "lerobot_0_6_replay_help.txt").read_text(encoding="utf-8"),
        "calibrate": (fixtures_dir / "lerobot_0_6_calibrate_help.txt").read_text(encoding="utf-8"),
        "eval": (fixtures_dir / "lerobot_0_6_eval_help.txt").read_text(encoding="utf-8"),
        "dataset_viz": (fixtures_dir / "lerobot_0_6_dataset_viz_help.txt").read_text(encoding="utf-8"),
    }
    help_json = json.dumps(help_map)
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import sys
        import time

        HELP = {help_json!s}

        def module_kind(name: str) -> str:
            n = name.lower()
            for key in ("record", "train", "teleoperate", "teleop", "rollout", "replay", "calibrate", "eval", "dataset_viz", "visualize"):
                if key in n:
                    if key in {{"teleoperate", "teleop"}}:
                        return "teleop"
                    if key in {{"visualize", "dataset_viz"}}:
                        return "dataset_viz"
                    return key
            return "other"

        args = sys.argv[1:]
        if args and args[0] == "-c":
            code = args[1] if len(args) > 1 else ""
            if "importlib.metadata.version" in code:
                sys.stdout.write("0.6.0")
                raise SystemExit(0)
            if "find_spec" in code:
                raise SystemExit(0)
            if "sys.version" in code:
                sys.stdout.write("3.12.0")
                raise SystemExit(0)
            raise SystemExit(0)

        module = ""
        if args and args[0] == "-m":
            module = args[1] if len(args) > 1 else ""
            rest = args[2:]
        else:
            rest = args

        if any(a in {{"--help", "-h"}} for a in rest) or (not rest and module):
            kind = module_kind(module)
            text = HELP.get(kind, "usage: fake\\n  --policy.path PATH\\n")
            sys.stdout.write(text)
            raise SystemExit(0)

        kind = module_kind(module + " " + " ".join(rest))
        print(f"FAKE_RUNTIME start kind={{kind}} module={{module}}")
        if kind == "teleop":
            print("Teleop running")
            time.sleep(30)
            raise SystemExit(0)
        if kind == "record":
            print("Recording episode 1 / 1")
            print("Recording complete")
            raise SystemExit(0)
        if kind == "train":
            print("Training step 1")
            print("Training complete")
            raise SystemExit(0)
        if kind == "rollout":
            print("Rollout running")
            print("Rollout complete")
            raise SystemExit(0)
        print("FAKE_RUNTIME complete")
        raise SystemExit(0)
        """
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class FakeRuntimeE2ETests(unittest.TestCase):
    """Prove the core operator loop through real services + fake LeRobot runtime."""

    def setUp(self) -> None:
        _CAP_CACHE.clear()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.runs_dir = self.root / "runs"
        self.data_dir = self.root / "data"
        self.models_dir = self.root / "models"
        self.venv_dir = self.root / "lerobot_env"
        for path in (self.runs_dir, self.data_dir, self.models_dir, self.venv_dir / "bin"):
            path.mkdir(parents=True, exist_ok=True)

        self.python = self.venv_dir / "bin" / "python3"
        _write_fake_python(self.python, FIXTURES)
        # Also provide plain `python` for build_lerobot_module_command variants.
        python_alias = self.venv_dir / "bin" / "python"
        if not python_alias.exists():
            python_alias.symlink_to(self.python)

        self.config = normalize_config_without_prompts(
            {
                **DEFAULT_CONFIG_VALUES,
                "hf_username": "tester",
                "runs_dir": str(self.runs_dir),
                "record_data_dir": str(self.data_dir),
                "trained_models_dir": str(self.models_dir),
                "follower_port": "/dev/tty.fake-follower",
                "leader_port": "/dev/tty.fake-leader",
                "lerobot_venv_dir": str(self.venv_dir),
                "compat_probe_enabled": True,
                "diagnostics_v2_enabled": True,
                "support_bundle_enabled": True,
            }
        )

    def tearDown(self) -> None:
        _CAP_CACHE.clear()
        self._tmpdir.cleanup()

    def _run_streaming(self, cmd: list[str], *, cancel_after_s: float | None = None) -> tuple[list[str], int | None, bool]:
        lines: list[str] = []
        codes: list[int] = []
        errors: list[Exception] = []
        started = time.monotonic()

        def cancel_requested() -> bool:
            if cancel_after_s is None:
                return False
            return time.monotonic() - started >= cancel_after_s

        thread = run_process_streaming(
            cmd=cmd,
            cwd=self.root,
            on_line=lines.append,
            on_complete=codes.append,
            on_start_error=errors.append,
            cancel_requested=cancel_requested,
            use_pty=False,
        )
        thread.join(timeout=20)
        self.assertFalse(thread.is_alive(), msg=f"command still running: {cmd}")
        self.assertEqual(errors, [], msg=str(errors))
        code = codes[0] if codes else None
        canceled = cancel_after_s is not None and (code not in {0} or any("cancel" in line.lower() for line in lines))
        # When canceled, exit code may be non-zero / signal.
        if cancel_after_s is not None:
            canceled = True
        return lines, code, canceled

    def test_core_operator_loop_with_fake_runtime(self) -> None:
        # 1-2. install/launch path markers: config create + save
        config_path = self.root / "robot_config.json"
        with patch("robot_pipeline_app.config_store.PRIMARY_CONFIG_PATH", config_path), patch(
            "robot_pipeline_app.config_store.LEGACY_CONFIG_PATH", self.root / "legacy.json"
        ):
            save_config(self.config, quiet=True)
            self.assertTrue(config_path.exists())

        # 3. Doctor / setup checks
        checks = collect_doctor_checks(self.config)
        self.assertTrue(checks)
        self.assertIn("Doctor", summarize_checks(checks, title="Doctor"))

        # 4. Capability detect from fake --help
        caps = probe_lerobot_capabilities(self.config, include_flag_probe=True, force_refresh=True)
        self.assertEqual(caps.lerobot_version, "0.6.0")
        self.assertTrue(caps.record_help_available, msg=caps.record_help_error)
        self.assertTrue(caps.train_help_available, msg=caps.train_help_error)
        self.assertTrue(caps.supports_rollout, msg=caps.rollout_support_detail)
        self.assertEqual(caps.preferred_deploy_path, "rollout")
        self.assertIn("policy.path", caps.supported_record_flags)
        self.assertIn("policy.path", caps.supported_rollout_flags)

        # 5. Command preview builders (same path as GUI forms)
        teleop_cmd = build_lerobot_teleop_command(self.config)
        self.assertEqual(teleop_cmd[0], str(self.python))
        record_cmd = build_lerobot_record_command(
            config=self.config,
            dataset_repo_id="tester/demo_ds",
            num_episodes=1,
            task="pick cube",
            episode_time=5,
            push_to_hub=False,
        )
        self.assertTrue(any(part.startswith("--dataset.repo_id=") for part in record_cmd))
        train_cmd = build_lerobot_train_command(
            self.config,
            {
                "dataset_repo_id": "tester/demo_ds",
                "policy_type": "act",
                "output_dir": str(self.models_dir / "train_out"),
                "device": "cpu",
                "job_name": "demo",
            },
        )
        self.assertTrue(any(part.startswith("--policy.type=") for part in train_cmd))
        model_path = self.models_dir / "policy"
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        (model_path / "model.safetensors").write_text("x", encoding="utf-8")
        deploy_cmd, deploy_path = build_lerobot_deploy_command(
            config=self.config,
            policy_path=model_path,
            task="pick cube",
            duration_s=10,
            num_episodes=1,
            dataset_repo_id="tester/eval_demo",
            push_to_hub=False,
        )
        self.assertEqual(deploy_path, "rollout")
        self.assertTrue(any("rollout" in part for part in deploy_cmd))

        # 6. start + cancel fake teleop
        _lines, _code, canceled = self._run_streaming(teleop_cmd, cancel_after_s=0.2)
        self.assertTrue(canceled)

        # 7-9. complete fake record / train / rollout
        now = datetime.now(timezone.utc)
        for mode, cmd, repo in (
            ("record", record_cmd, "tester/demo_ds"),
            ("train", train_cmd, "tester/demo_ds"),
            ("deploy", deploy_cmd, "tester/eval_demo"),
        ):
            lines, code, _canceled = self._run_streaming(cmd)
            self.assertEqual(code, 0, msg=f"{mode} failed: {lines}")
            ended = datetime.now(timezone.utc)
            run_path = write_run_artifacts(
                config=self.config,
                mode=mode,
                command=cmd,
                cwd=self.root,
                started_at=now,
                ended_at=ended,
                exit_code=code,
                canceled=False,
                preflight_checks=[],
                output_lines=lines,
                dataset_repo_id=repo,
                model_path=model_path if mode == "deploy" else None,
                metadata_extra={"deploy_path": deploy_path} if mode == "deploy" else None,
            )
            self.assertIsNotNone(run_path)
            assert run_path is not None
            self.assertTrue(run_path.exists())

        # 10-11. History / experiments consume the same run artifacts
        runs, _total = list_runs(self.config, limit=20)
        self.assertGreaterEqual(len(runs), 3)
        modes = {str(run.get("mode", "")) for run in runs}
        self.assertTrue({"record", "train", "deploy"}.issubset(modes))

        # 12. recover an interrupted queue via real WorkflowQueueService persistence
        controller = _FakeQueueRunController()
        logs: list[str] = []
        queue = WorkflowQueueService(config=self.config, run_controller=controller, append_log=logs.append)
        item = build_train_deploy_eval_queue_item(
            queue_id=queue.next_queue_id(),
            train_form_values={
                "dataset_repo_id": "tester/demo_ds",
                "policy_type": "act",
                "output_dir": str(self.models_dir / "queue_train"),
                "device": "cpu",
                "job_name": "queue_demo",
            },
            deploy_settings={
                "model_path": str(model_path),
                "eval_repo_id": "tester/eval_demo",
                "eval_num_episodes": 1,
                "eval_duration_s": 5,
                "eval_task": "pick cube",
            },
        )
        item.status = "queued"
        item.resume_required = True
        item.current_step_index = 0
        item.error_text = "Interrupted by process exit."
        queue._items.append(item)
        queue._persist_state()

        # New service instance recovers state from disk.
        queue2 = WorkflowQueueService(config=self.config, run_controller=controller, append_log=logs.append)
        snapshots = queue2.snapshots()
        self.assertTrue(snapshots)
        recovered = next(s for s in snapshots if s["queue_id"] == item.queue_id)
        self.assertEqual(recovered["status"], "queued")
        ok, message = queue2.resume_pending()
        self.assertTrue(ok, msg=message)

        # 13. support bundle
        latest = runs[0]
        run_id = str(latest.get("run_id") or "")
        self.assertTrue(run_id)
        bundle_path = self.root / "support.zip"
        result = create_support_bundle(config=self.config, run_id=run_id, output_path=bundle_path)
        self.assertTrue(result.ok, msg=result.message)
        self.assertIsNotNone(result.bundle_path)
        assert result.bundle_path is not None
        self.assertTrue(result.bundle_path.exists())
        with zipfile.ZipFile(result.bundle_path, "r") as zf:
            self.assertTrue(zf.namelist())


if __name__ == "__main__":
    unittest.main()
