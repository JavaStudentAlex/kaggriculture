#!/usr/bin/env python3
"""Direct launcher for Kaggriculture Shinka evolution with reasoning effort injection.

Borrowed from the vesselDetector shinka pattern:
Standard Shinka drops reasoning effort for local_openai models (sending only max_tokens
and temperature). This runner intercepts the local_reasoning_effort block from
shinka_config.yaml and wraps shinka.llm.llm.sample_model_kwargs so that every local LLM
proposal and meta-supervision call passes reasoning_effort (e.g. 'xhigh' or 'high')
directly to the local proxy.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml
from shinka.cli import run as sc_run
from shinka.cli import run_config as rc
from shinka.llm import llm as shinka_llm
from shinka.llm.providers.model_resolver import resolve_model_backend


def apply_local_reasoning_effort(efforts: dict[str, Any]) -> None:
    """Wrap sample_model_kwargs to attach reasoning_effort for local_openai models."""
    if not efforts:
        return
    efforts = dict(efforts)
    default_effort = efforts.pop("default", None)
    original_sampler = shinka_llm.sample_model_kwargs

    def sample_with_effort(*args: Any, **kwargs: Any) -> dict[str, Any]:
        sampled = original_sampler(*args, **kwargs)
        model_name = sampled.get("model_name")
        if not model_name:
            return sampled
        try:
            resolved = resolve_model_backend(model_name)
        except ValueError:
            return sampled
        if resolved.provider == "local_openai":
            effort = efforts.get(resolved.api_model_name, default_effort)
            if effort is not None:
                sampled["reasoning_effort"] = str(effort)
        return sampled

    shinka_llm.sample_model_kwargs = sample_with_effort
    print(
        f"[run_evo] Active local reasoning effort: default={default_effort}, "
        f"model overrides={efforts}"
    )


_orig_load_optional_yaml_config = rc.load_optional_yaml_config


def _patched_load_optional_yaml_config(
    *,
    task_dir: Path,
    config_fname: str | None,
    allowed_field_types: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Extract local_reasoning_effort before validating against Shinka's schema."""
    if not config_fname:
        return _orig_load_optional_yaml_config(
            task_dir=task_dir,
            config_fname=config_fname,
            allowed_field_types=allowed_field_types,
        )

    config_path = Path(config_fname)
    if not config_path.is_absolute():
        config_path = task_dir / config_path
    if not config_path.is_file():
        return _orig_load_optional_yaml_config(
            task_dir=task_dir,
            config_fname=config_fname,
            allowed_field_types=allowed_field_types,
        )

    raw_text = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, dict):
        return _orig_load_optional_yaml_config(
            task_dir=task_dir,
            config_fname=config_fname,
            allowed_field_types=allowed_field_types,
        )

    local_efforts = loaded.pop("local_reasoning_effort", None)
    if local_efforts is not None:
        apply_local_reasoning_effort(local_efforts)

    # Dump the sanitized config to a temporary file so shinka's strict key validation passes
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=True) as tmp:
        yaml.safe_dump(loaded, tmp)
        tmp.flush()
        return _orig_load_optional_yaml_config(
            task_dir=Path(tmp.name).parent,
            config_fname=Path(tmp.name).name,
            allowed_field_types=allowed_field_types,
        )


def main(argv: list[str] | None = None) -> int:
    rc.load_optional_yaml_config = _patched_load_optional_yaml_config
    return sc_run.main(argv)


if __name__ == "__main__":
    sys.exit(main())
