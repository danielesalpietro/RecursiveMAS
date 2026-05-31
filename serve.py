#!/usr/bin/env python3
"""
Gradio web UI for RecursiveMAS.

Models are loaded once on first use and kept warm in VRAM.
Style switching evicts the old models before loading the new set.

Launch:
    python serve.py                          # http://0.0.0.0:7860
    python serve.py --port 8080 --share      # public Gradio link
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import queue as _queue
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Load .env (TAVILY_API_KEY needed for deliberation style)
_env_path = THIS_DIR / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")

# ── Model cache — patch BEFORE importing any pipeline module ─────────────────
# All submodules import inference_mas as `base` and call base.load_agent_model_and_tokenizer
# at call time (not import time), so patching the module attribute propagates correctly.

import inference_utils.inference_mas as _base  # noqa: E402
from prompts import DOMAIN_SYSTEM_PROMPTS, set_active_domain  # noqa: E402

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_CURRENT_STYLE: Optional[str] = None
_ORIG_LOAD = _base.load_agent_model_and_tokenizer


def _cached_load(model_name_or_path, device, dtype, trust_remote_code, agent_name):
    from modeling import resolve_local_pretrained_path
    key = resolve_local_pretrained_path(str(model_name_or_path))
    if key not in _MODEL_CACHE:
        print(f"[serve] loading {agent_name} into VRAM …", flush=True)
        model, tok = _ORIG_LOAD(model_name_or_path, device, dtype, trust_remote_code, agent_name)
        _MODEL_CACHE[key] = (model, tok)
        print(f"[serve] {agent_name} cached.", flush=True)
    else:
        print(f"[serve] cache hit: {agent_name}", flush=True)
    return _MODEL_CACHE[key]


def _noop_release(*_):
    pass  # keep models warm between requests


_base.load_agent_model_and_tokenizer = _cached_load
_base.release_resources = _noop_release

# ── Import run.py utilities (safe now that patching is done) ─────────────────
from run import (  # noqa: E402
    STYLE_SPECS,
    build_cli_for_style,
    infer_max_new_tokens,
    resolve_style_paths,
)
import gradio as gr  # noqa: E402

_VERSION = (THIS_DIR / "VERSION").read_text(encoding="utf-8").strip()

# ── VRAM management ───────────────────────────────────────────────────────────

def _evict_cache() -> None:
    global _MODEL_CACHE
    for model, tok in _MODEL_CACHE.values():
        del model, tok
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[serve] VRAM cache evicted.", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_single_question(
    style: str,
    question: str,
    device: str,
    num_rounds: int,
    latent_steps: int,
    domain: str = "general",
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
    device_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    Run the MAS pipeline on one question.
    Returns (captured_stdout, parsed_answer_string).
    """
    import argparse as _ap

    set_active_domain(domain)

    # Write question to a temporary medqa-format JSON
    tmp_json = tempfile.mktemp(suffix=".json")
    result_jsonl = tempfile.mktemp(suffix=".jsonl")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump([{"question": question, "answer": ""}], f)

    try:
        # Resolve HF model paths (cached locally after first download)
        paths = resolve_style_paths(style, "math500")
        family = str(STYLE_SPECS[style]["family"])
        max_new_tokens = infer_max_new_tokens(style, "math500")

        # Minimal args namespace consumed by build_cli_for_style
        fake_args = _ap.Namespace(
            dataset="math500",
            dataset_split="",
            num_recursive_rounds=num_rounds,
            batch_size=1,
            latent_length=latent_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            trust_remote_code=1,
            device=device,
            seed=seed,
            sample_seed=seed,
        )

        module, cli_args = build_cli_for_style(
            args=fake_args,
            family=family,
            dataset_arg=tmp_json,   # actual dataset = temp JSON (single question)
            dataset_split="train",
            paths=paths,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            device_map=device_map or {},
        )
        cli_args += ["--result_jsonl", result_jsonl, "--num_samples", "-1"]

        # Run with stdout captured (also tee to global log buffer)
        class _TeeCapture:
            def __init__(self) -> None:
                self._buf = io.StringIO()
            def write(self, s: str) -> None:
                self._buf.write(s)
                _log_write(s)
            def flush(self) -> None:
                pass
            def getvalue(self) -> str:
                return self._buf.getvalue()

        captured = _TeeCapture()
        old_argv, sys.argv = sys.argv[:], [module.__file__ or "serve"] + cli_args
        try:
            with contextlib.redirect_stdout(captured):
                module.main()
        finally:
            sys.argv = old_argv

        stdout = captured.getvalue()

        # Read structured result
        parsed = ""
        if os.path.isfile(result_jsonl):
            with open(result_jsonl, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    parsed = str(
                        rec.get("pred_answer_parsed")
                        or rec.get("pred_code_parsed")
                        or ""
                    )
                    break

        return stdout, parsed

    finally:
        for p in (tmp_json, result_jsonl):
            try:
                os.unlink(p)
            except OSError:
                pass


def _parse_agent_outputs(stdout: str) -> Dict[str, str]:
    """Extract Agent1 / Agent2 / Agent3 text outputs from captured stdout."""
    # The pipeline prints labelled sections like "3) Agent1 Output:" etc.
    MARKERS = {
        "Agent1 Output:": "agent1",
        "Agent2 Output:": "agent2",
        "Agent3 Output:": "agent3",
    }
    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    buf: List[str] = []

    for line in stdout.splitlines():
        matched = False
        for marker, key in MARKERS.items():
            if marker in line:
                if current_key and buf:
                    sections[current_key] = "\n".join(buf).strip()
                current_key = key
                buf = []
                matched = True
                break
        if matched:
            continue
        if current_key:
            # Section ends at a separator or a new numbered label
            if line.startswith("=" * 20) or (
                len(line) > 2 and line[0].isdigit() and line[1] in (").", ") ")
            ):
                sections[current_key] = "\n".join(buf).strip()
                current_key = None
                buf = []
            else:
                buf.append(line)

    if current_key and buf:
        sections[current_key] = "\n".join(buf).strip()

    return sections


def _build_reply(
    style: str,
    parsed: str,
    stdout: str,
    run_info: Optional[Dict] = None,
) -> str:
    agents = _parse_agent_outputs(stdout)
    parts: List[str] = [f"**Style:** `{style}`"]

    if parsed:
        parts.append(f"\n**Answer:** `{parsed}`")

    solver_text = agents.get("agent3", "")
    if solver_text:
        parts.append("\n---\n**Solver output:**\n" + solver_text)

    # Wrap intermediate agent outputs in collapsible details
    for key, label in [("agent1", "Planner"), ("agent2", "Critic / Refiner")]:
        text = agents.get(key, "")
        if text:
            parts.append(
                f"\n<details><summary>{label} output</summary>\n\n{text}\n\n</details>"
            )

    if run_info:
        info = (
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Version | `v{run_info['version']}` |\n"
            f"| Style | `{run_info['style']}` |\n"
            f"| Domain | `{run_info['domain']}` |\n"
            f"| Recursive rounds | {run_info['rounds']} |\n"
            f"| Latent steps | {run_info['latent_steps']} |\n"
            f"| Temperature | {run_info['temperature']} |\n"
            f"| Top-p | {run_info['top_p']} |\n"
            f"| Seed | {run_info['seed']} |\n"
            f"| Device | `{run_info['device']}` |\n"
            f"| Started | {run_info['started']} |\n"
            f"| Finished | {run_info['finished']} |\n"
            f"| Elapsed | {run_info['elapsed']} |"
        )
        parts.append(f"\n<details><summary>Run info</summary>\n\n{info}\n\n</details>")

    return "\n".join(parts) if len(parts) > 1 else (parsed or stdout[:3000])


# ── Gradio event handler ──────────────────────────────────────────────────────

def respond(
    message: str,
    history: List[Dict],
    style: str,
    domain: str,
    num_rounds: int,
    latent_steps: int,
    device: str,
    temperature: float,
    top_p: float,
    seed: int,
    device_map_state: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict], str]:
    global _CURRENT_STYLE

    if not message.strip():
        return history, history, ""

    if _BATCH_LOCK.locked():
        warning = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ A batch evaluation is running. Please wait for it to finish before sending chat messages."},
        ]
        return history + warning, history + warning, ""

    if _CURRENT_STYLE != style and _MODEL_CACHE:
        _evict_cache()
    _CURRENT_STYLE = style

    device_map = (device_map_state or {}).get(style)

    try:
        t_start = datetime.now()
        stdout, parsed = _run_single_question(
            style, message, device, num_rounds, latent_steps, domain,
            temperature=temperature, top_p=top_p, seed=seed,
            device_map=device_map,
        )
        t_end = datetime.now()
        elapsed = str(t_end - t_start).split(".")[0]  # HH:MM:SS
        run_info = {
            "version": _VERSION,
            "style": style,
            "domain": domain,
            "rounds": num_rounds,
            "latent_steps": latent_steps,
            "device": device,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "started": t_start.strftime("%Y-%m-%d %H:%M:%S"),
            "finished": t_end.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": elapsed,
        }
        reply = _build_reply(style, parsed, stdout, run_info)
    except Exception as exc:
        reply = f"❌ Error during inference:\n```\n{exc}\n```"

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, new_history, ""


# ── Batch evaluation backend ──────────────────────────────────────────────────

_BATCH_LOCK = threading.Lock()   # prevents concurrent batch runs
_BATCH_STOP_EVENT = threading.Event()  # set to request cooperative stop
_BATCH_DATASETS = ["math500", "medqa", "gpqa", "mbppplus"]


class _BatchStopped(Exception):
    pass

# inference_mas.py defaults dataset_split to "test", but build_common_cli in
# run.py overrides it with "" (empty), which HuggingFace rejects.  Map each
# dataset to the split it actually uses so we can pass it explicitly.
# medqa and mbppplus use local code paths and ignore the split value entirely.
_DATASET_SPLITS: Dict[str, str] = {
    "math500": "test",    # HuggingFaceH4/MATH-500 → test split
    "medqa":   "train",   # local __local_medqa__ path — value not used
    "gpqa":    "train",   # Idavidrein/gpqa gpqa_diamond → train split
    "mbppplus": "test",   # __mbppplus__ local path — value not used
}


class _QueueWriter:
    """Redirect stdout from the inference thread into a Queue.

    Checks _BATCH_STOP_EVENT on every write so that clicking Stop
    terminates the batch at the next print call inside the pipeline.
    """
    def __init__(self, q: "_queue.Queue[Optional[str]]") -> None:
        self._q = q

    def write(self, s: str) -> None:
        if _BATCH_STOP_EVENT.is_set():
            raise _BatchStopped()
        if s:
            self._q.put(s)
            _log_write(s)

    def flush(self) -> None:
        pass


def _batch_worker(
    style: str,
    dataset: str,
    num_samples: int,
    device: str,
    rounds: int,
    latent_steps: int,
    temperature: float,
    top_p: float,
    seed: int,
    result_jsonl: str,
    log_q: "_queue.Queue[Optional[str]]",
    device_map: Optional[Dict[str, str]] = None,
) -> None:
    """Runs the MAS pipeline over a full dataset in a background thread."""
    import argparse as _ap

    try:
        paths = resolve_style_paths(style, dataset)
        family = str(STYLE_SPECS[style]["family"])
        max_new_tokens = infer_max_new_tokens(style, dataset)

        fake_args = _ap.Namespace(
            dataset=dataset,
            dataset_split="",
            num_recursive_rounds=rounds,
            batch_size=8,
            latent_length=latent_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            trust_remote_code=1,
            device=device,
            seed=seed,
            sample_seed=-1,
        )

        dataset_split = _DATASET_SPLITS.get(dataset.lower(), "test")
        module, cli_args = build_cli_for_style(
            args=fake_args,
            family=family,
            dataset_arg=dataset,
            dataset_split=dataset_split,
            paths=paths,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            device_map=device_map or {},
        )

        cli_args += ["--result_jsonl", result_jsonl]
        if num_samples > 0:
            # build_common_cli already added --num_samples -1; argparse last-wins
            cli_args += ["--num_samples", str(num_samples)]

        old_argv = sys.argv[:]
        sys.argv = [module.__file__ or "batch"] + cli_args
        try:
            with contextlib.redirect_stdout(_QueueWriter(log_q)):
                module.main()
            log_q.put(None)  # sentinel: success
        except _BatchStopped:
            log_q.put("\n⏹ Batch stopped by user.\n")
            log_q.put(None)
        except Exception as exc:
            log_q.put(f"\n❌ Runtime error: {exc}\n")
            log_q.put(None)
        finally:
            sys.argv = old_argv

    except Exception as exc:
        log_q.put(f"\n❌ Setup error: {exc}\n")
        log_q.put(None)


def stop_batch_eval() -> None:
    """Signal the running batch worker to stop at its next print call."""
    _BATCH_STOP_EVENT.set()


def run_batch_eval(
    style: str,
    dataset: str,
    num_samples: int,
    device: str,
    rounds: int,
    latent_steps: int,
    temperature: float,
    top_p: float,
    seed: int,
    device_map_state: Optional[Dict] = None,
):
    """Gradio generator: streams log lines and yields (log_text, file, run_btn, stop_btn)."""
    import re as _re

    _btn_running = (gr.update(interactive=False), gr.update(interactive=True))
    _btn_idle    = (gr.update(interactive=True),  gr.update(interactive=False))

    if not _BATCH_LOCK.acquire(blocking=False):
        yield ("⚠️ A batch run is already in progress. Wait for it to finish.\n",
               gr.update(visible=False), *_btn_idle)
        return

    _BATCH_STOP_EVENT.clear()

    device_map = (device_map_state or {}).get(style)
    result_jsonl = tempfile.mktemp(suffix=".jsonl", prefix="house_batch_")
    log_q: "_queue.Queue[Optional[str]]" = _queue.Queue()

    thread = threading.Thread(
        target=_batch_worker,
        args=(style, dataset, num_samples, device, rounds, latent_steps,
              temperature, top_p, seed, result_jsonl, log_q, device_map),
        daemon=True,
    )
    thread.start()

    n_label = "all" if num_samples <= 0 else str(num_samples)
    log = (
        f"🚀 Batch evaluation started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"   style={style} | dataset={dataset} | samples={n_label}\n"
        f"   rounds={rounds} | latent_steps={latent_steps} | temperature={temperature}"
        f" | top_p={top_p} | seed={seed} | version=v{_VERSION}\n"
        f"{'='*70}\n"
    )
    yield log, gr.update(visible=False), *_btn_running

    while True:
        try:
            item = log_q.get(timeout=1.0)
        except _queue.Empty:
            yield log, gr.update(visible=False), *_btn_running
            continue
        if item is None:
            break
        log += item
        yield log, gr.update(visible=False), *_btn_running

    thread.join()
    _BATCH_LOCK.release()

    was_stopped = _BATCH_STOP_EVENT.is_set()
    if was_stopped:
        summary = f"\n{'='*70}\n⏹ Stopped — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    else:
        acc_matches = _re.findall(r"accuracy=([0-9]+(?:\.[0-9]+)?)%", log)
        summary = f"\n{'='*70}\n✅ Batch complete — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if acc_matches:
            summary += f" | accuracy: {acc_matches[-1]}%"
        summary += "\n"
    log += summary

    if not was_stopped and os.path.isfile(result_jsonl):
        yield log, gr.update(value=result_jsonl, visible=True), *_btn_idle
    else:
        yield log, gr.update(visible=False), *_btn_idle


# ── Model Manager backend ─────────────────────────────────────────────────────

_DOWNLOAD_LOCK = threading.Lock()


def _all_model_repos() -> List[Tuple[str, str, str]]:
    """Return list of (style_name, role, repo_id) for all managed models."""
    result = []
    for style_name, spec in STYLE_SPECS.items():
        for role, repo_id in spec["repos"].items():
            result.append((style_name, role, str(repo_id)))
    return result


def _get_cached_repos() -> Dict[str, int]:
    """Return {repo_id: size_bytes} for repos present in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        return {r.repo_id: r.size_on_disk for r in cache_info.repos}
    except Exception:
        return {}


def _fmt_size(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


def _build_model_catalog_data() -> List[List]:
    cached = _get_cached_repos()
    rows = []
    for style_name, role, repo_id in _all_model_repos():
        if repo_id in cached:
            status = "✅ cached"
            size_str = _fmt_size(cached[repo_id])
        else:
            status = "☁️ not downloaded"
            size_str = "—"
        rows.append([False, style_name, role, repo_id, status, size_str])
    return rows


def _fetch_remote_info(repo_id: str) -> Tuple[Optional[str], Optional[int]]:
    """Query HF Hub for latest commit SHA and total expected size (network call)."""
    try:
        from huggingface_hub import model_info as _hf_model_info, list_repo_tree as _hf_list_tree
        info = _hf_model_info(repo_id)
        remote_sha: Optional[str] = info.sha
        total = sum(
            getattr(f, "size", None) or 0
            for f in _hf_list_tree(repo_id, recursive=True, repo_type="model")
            if hasattr(f, "size")
        )
        return remote_sha, total if total > 0 else None
    except Exception:
        return None, None


def _download_worker(repo_id: str, log_q: "_queue.Queue[Optional[str]]") -> None:
    try:
        from huggingface_hub import snapshot_download
        log_q.put(f"📥 Downloading {repo_id} …\n")
        local_path = snapshot_download(repo_id=repo_id)
        log_q.put(f"✅ Done — cached at: {local_path}\n")
    except Exception as exc:
        log_q.put(f"❌ Download failed: {exc}\n")
    finally:
        log_q.put(None)  # sentinel


def _mm_run_action(action: Optional[str], df_data):
    """Gradio generator: run selected action on all checked rows, one at a time."""
    if not action:
        yield "⚠️ Choose an action from the dropdown.", gr.update()
        return

    try:
        import pandas as _pd
        rows = df_data.values.tolist() if isinstance(df_data, _pd.DataFrame) else (df_data or [])
    except ImportError:
        rows = df_data or []

    # col layout: [bool, style, role, repo_id, status, size]
    selected = [
        r[3] for r in rows
        if r and (r[0] is True or r[0] == "True" or r[0] == 1)
    ]

    if not selected:
        yield "⚠️ No models selected — tick the checkboxes first.", gr.update()
        return

    # ── Download / Update ─────────────────────────────────────────────────────
    if "Download" in action or "Update" in action:
        verb = "Updating" if "Update" in action else "Downloading"
        log = f"⬇ {verb} {len(selected)} model(s)…\n\n"
        yield log, gr.update()

        if not _DOWNLOAD_LOCK.acquire(blocking=False):
            yield log + "⚠️ Another download is already running.", gr.update()
            return
        try:
            for repo_id in selected:
                log_q: "_queue.Queue[Optional[str]]" = _queue.Queue()
                t = threading.Thread(target=_download_worker, args=(repo_id, log_q), daemon=True)
                t.start()
                while True:
                    try:
                        item = log_q.get(timeout=2.0)
                    except _queue.Empty:
                        yield log, gr.update()
                        continue
                    if item is None:
                        break
                    log += item
                    yield log, gr.update()
                t.join()
        finally:
            _DOWNLOAD_LOCK.release()

        log += "\n✅ Done.\n"
        yield log, gr.update(value=_build_model_catalog_data())
        return

    # ── Check ─────────────────────────────────────────────────────────────────
    if "Check" in action:
        log = f"🔍 Checking {len(selected)} model(s) against HF Hub…\n\n"
        yield log, gr.update()

        try:
            from huggingface_hub import scan_cache_dir
            cached_map = {r.repo_id: r for r in scan_cache_dir().repos}
        except Exception:
            cached_map = {}

        # Mutable copy of the current table; we'll patch Status/Size inline
        current_rows = [list(r) for r in rows]
        row_by_repo = {r[3]: r for r in current_rows}

        for repo_id in selected:
            log += f"· {repo_id}\n"
            yield log, current_rows

            remote_sha, remote_size = _fetch_remote_info(repo_id)
            row = row_by_repo.get(repo_id)

            if remote_sha is None:
                log += "  ❌ Could not reach HF Hub\n"
                if row:
                    row[4] = "❌ check failed"
                yield log, current_rows
                continue

            if repo_id not in cached_map:
                log += "  ☁️ Not cached locally\n"
                if row:
                    row[4] = "☁️ not downloaded"
                yield log, current_rows
                continue

            local_repo = cached_map[repo_id]
            local_size = local_repo.size_on_disk

            local_sha: Optional[str] = None
            if local_repo.revisions:
                try:
                    latest = max(local_repo.revisions, key=lambda rv: rv.last_modified)
                except Exception:
                    latest = next(iter(local_repo.revisions))
                local_sha = latest.commit_hash

            if remote_size and local_size < remote_size * 0.85:
                pct = local_size / remote_size * 100
                msg = f"⚠️ incomplete ({pct:.0f}% of {_fmt_size(remote_size)})"
                log += f"  {msg}\n"
                if row:
                    row[4] = msg
                    row[5] = f"{_fmt_size(local_size)} / {_fmt_size(remote_size)}"
            elif local_sha and remote_sha and local_sha != remote_sha:
                msg = f"🔄 update available ({local_sha[:7]}→{remote_sha[:7]})"
                log += f"  {msg}\n"
                if row:
                    row[4] = msg
            else:
                short = local_sha[:7] if local_sha else "?"
                msg = f"✅ up to date ({short})"
                log += f"  {msg}\n"
                if row:
                    row[4] = msg

            yield log, current_rows

        log += "\n✅ Check complete.\n"
        yield log, current_rows
        return

    # ── Delete ────────────────────────────────────────────────────────────────
    if "Delete" in action:
        log = f"🗑 Deleting {len(selected)} model(s)…\n\n"
        yield log, gr.update()

        for repo_id in selected:
            try:
                from huggingface_hub import scan_cache_dir
                cache_info = scan_cache_dir()
                target = next((r for r in cache_info.repos if r.repo_id == repo_id), None)
                if target is None:
                    log += f"⚠️ `{repo_id}` not found in cache.\n"
                else:
                    hashes = [rev.commit_hash for rev in target.revisions]
                    cache_info.delete_revisions(*hashes).execute()
                    log += f"🗑️ Deleted `{repo_id}`\n"
            except Exception as exc:
                log += f"❌ Failed `{repo_id}`: {exc}\n"
            yield log, gr.update()

        log += "\n✅ Done.\n"
        yield log, gr.update(value=_build_model_catalog_data())


# ── Architecture descriptions ─────────────────────────────────────────────────

_STYLE_DESCRIPTIONS: Dict[str, str] = {
    "sequential_light": (
        "**Planner → Critic → Solver** via latent recursion (~5 GB VRAM).  \n"
        "Lightweight pipeline — ideal for math and step-by-step reasoning."
    ),
    "sequential_scaled": (
        "**Planner → Critic → Solver** with larger models (~12 GB VRAM).  \n"
        "Same pipeline as Light but higher accuracy on complex tasks."
    ),
    "mixture": (
        "**Math + Code + Science specialists → Summarizer** in parallel (~15 GB VRAM).  \n"
        "Best for questions that span multiple domains."
    ),
    "distillation": (
        "**Expert → Learner** with bidirectional latent feedback (~18 GB VRAM).  \n"
        "Strong knowledge transfer for hard reasoning problems."
    ),
    "deliberation": (
        "**Reflector → Toolcaller + external tools** (~12 GB VRAM).  \n"
        "Best for tasks that benefit from web search or code execution."
    ),
}

_ARCH_HTML = """
<style>
  .aw{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:8px 4px}
  .ab{margin-bottom:28px}
  .ab h3{margin:0 0 5px;font-size:1.1em;color:#1a1a2e}
  .ad{color:#555;font-size:.88em;margin:0 0 10px;line-height:1.5}
  .pl{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:6px}
  .ag{background:#e8f4fd;border:2px solid #2196F3;border-radius:8px;padding:7px 12px;text-align:center;min-width:100px}
  .ag .m{font-size:.72em;color:#777}
  .ag .r{font-weight:700;color:#1565C0;font-size:.9em}
  .ag.ex{background:#e8f5e9;border-color:#4CAF50}
  .ag.ex .r{color:#2e7d32}
  .ag.sp{background:#f3e5f5;border-color:#9C27B0}
  .ag.sp .r{color:#6a1b9a}
  .ag.out{background:#fff8e1;border:2px solid #FFC107}
  .ag.out .r{color:#e65100}
  .rl{background:#fff3e0;border:2px dashed #FF9800;border-radius:6px;padding:3px 8px;
      font-size:.72em;color:#bf360c;white-space:nowrap;font-weight:700}
  .ar{font-size:1.3em;color:#bbb}
  .fb{font-size:.78em;color:#999;font-style:italic;margin-top:3px}
  .pg{display:flex;flex-direction:column;gap:6px}
  .rg{display:flex;flex-direction:column;gap:6px;align-items:center;justify-content:center}
  .tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
  .tag{background:#f0f0f0;border-radius:12px;padding:2px 9px;font-size:.76em;color:#555}
  hr.s{border:none;border-top:1px solid #eee;margin:22px 0}
  .legend{font-size:.77em;color:#aaa;margin-top:14px;line-height:1.6}
</style>

<div class="aw">

<!-- Sequential Light -->
<div class="ab">
  <h3>🔗 Sequential Light</h3>
  <p class="ad">A lightweight 3-agent pipeline. The <b>Planner</b> outlines a strategy, the <b>Critic</b>
  reviews and refines it, and the <b>Solver</b> produces the final answer — all via latent-space tensors,
  never text. In multi-round mode the Solver's latent feeds back to the Planner.</p>
  <div class="pl">
    <div class="ag"><div class="m">Qwen3-1.7B</div><div class="r">Planner</div></div>
    <span class="rl">→ RL →</span>
    <div class="ag"><div class="m">Llama3.2-1B</div><div class="r">Critic</div></div>
    <span class="rl">→ RL →</span>
    <div class="ag"><div class="m">Qwen2.5-Math-1.5B</div><div class="r">Solver</div></div>
    <span class="ar">→</span>
    <div class="ag out"><div class="m">&nbsp;</div><div class="r">Answer</div></div>
  </div>
  <div class="fb">↺ multi-round: Solver latent → RL → Planner (next round)</div>
  <div class="tags"><span class="tag">≈ 5 GB VRAM</span><span class="tag">Math · Reasoning</span><span class="tag">~3.2B total params</span></div>
</div>
<hr class="s">

<!-- Sequential Scaled -->
<div class="ab">
  <h3>🔗 Sequential Scaled</h3>
  <p class="ad">Same pipeline architecture as Sequential Light, but with larger models for higher accuracy at the cost of more VRAM.</p>
  <div class="pl">
    <div class="ag"><div class="m">Gemma3-4B</div><div class="r">Planner</div></div>
    <span class="rl">→ RL →</span>
    <div class="ag"><div class="m">Llama3.2-3B</div><div class="r">Critic</div></div>
    <span class="rl">→ RL →</span>
    <div class="ag"><div class="m">Qwen3.5-4B</div><div class="r">Solver</div></div>
    <span class="ar">→</span>
    <div class="ag out"><div class="m">&nbsp;</div><div class="r">Answer</div></div>
  </div>
  <div class="fb">↺ multi-round: Solver latent → RL → Planner (next round)</div>
  <div class="tags"><span class="tag">≈ 12 GB VRAM</span><span class="tag">Math · Complex reasoning</span><span class="tag">~11B total params</span></div>
</div>
<hr class="s">

<!-- Mixture -->
<div class="ab">
  <h3>🌐 Mixture</h3>
  <p class="ad">Three domain specialists run <em>in parallel</em>. Each produces a latent embedding of its
  analysis; the <b>Summarizer</b> receives all three simultaneously and synthesises the final answer.
  In multi-round mode the Summarizer sends individual feedback latents back to each specialist.</p>
  <div class="pl">
    <div class="pg">
      <div class="ag sp"><div class="m">DeepSeek-R1-Qwen-1.5B</div><div class="r">Math</div></div>
      <div class="ag sp"><div class="m">Qwen2.5-Coder-3B</div><div class="r">Code</div></div>
      <div class="ag sp"><div class="m">BioMistral-7B</div><div class="r">Science</div></div>
    </div>
    <div class="rg">
      <span class="rl">→ RL →</span>
      <span class="rl">→ RL →</span>
      <span class="rl">→ RL →</span>
    </div>
    <div class="ag ex"><div class="m">Qwen3.5-2B</div><div class="r">Summarizer</div></div>
    <span class="ar">→</span>
    <div class="ag out"><div class="m">&nbsp;</div><div class="r">Answer</div></div>
  </div>
  <div class="fb">↺ multi-round: Summarizer latent → RL → each Specialist (next round)</div>
  <div class="tags"><span class="tag">≈ 15 GB VRAM</span><span class="tag">Multi-domain · Science · Code</span><span class="tag">~13.5B total params</span></div>
</div>
<hr class="s">

<!-- Distillation -->
<div class="ab">
  <h3>🎓 Distillation</h3>
  <p class="ad">The <b>Expert</b> reasons deeply and transmits a compressed latent representation to the
  <b>Learner</b>, which produces the final answer. In multi-round mode the Learner feeds a latent signal
  back to the Expert, enabling iterative knowledge transfer in both directions.</p>
  <div class="pl">
    <div class="ag ex"><div class="m">Qwen3.5-9B</div><div class="r">Expert</div></div>
    <span class="rl">→ RL_el →</span>
    <div class="ag"><div class="m">Qwen3.5-4B</div><div class="r">Learner</div></div>
    <span class="ar">→</span>
    <div class="ag out"><div class="m">&nbsp;</div><div class="r">Answer</div></div>
  </div>
  <div class="fb">↺ multi-round: Learner latent → RL_le → Expert (next round)</div>
  <div class="tags"><span class="tag">≈ 18 GB VRAM</span><span class="tag">Knowledge transfer · Hard reasoning</span><span class="tag">~13B total params</span></div>
</div>
<hr class="s">

<!-- Deliberation -->
<div class="ab">
  <h3>🔭 Deliberation</h3>
  <p class="ad">The <b>Reflector</b> analyses the problem and passes a latent signal to the <b>Toolcaller</b>,
  which can invoke external tools (web search, Python interpreter) before producing the final answer.
  In multi-round mode the Toolcaller's state feeds back to the Reflector.</p>
  <div class="pl">
    <div class="ag"><div class="m">Qwen3.5-4B</div><div class="r">Reflector</div></div>
    <span class="rl">→ RL_rt →</span>
    <div class="ag sp"><div class="m">Qwen3.5-4B</div><div class="r">Toolcaller</div></div>
    <span class="ar">→</span>
    <div class="ag out"><div class="m">🌐 🐍</div><div class="r">Tools + Answer</div></div>
  </div>
  <div class="fb">↺ multi-round: Toolcaller latent → RL_tr → Reflector (next round)</div>
  <div class="tags"><span class="tag">≈ 12 GB VRAM</span><span class="tag">Tool use · Web search · Code exec</span><span class="tag">~8B total params</span></div>
</div>

<div class="legend">
  <b>RL</b> = RecursiveLink — a small trained adapter (MLP) that transforms hidden-state tensors between
  different model architectures. All inter-agent communication within a round happens entirely in latent
  space; no text is exchanged between agents.
</div>

</div>
"""

# ── Multi-GPU helpers ─────────────────────────────────────────────────────────

_STYLE_AGENT_ROLES: Dict[str, List[Tuple[str, str]]] = {
    "sequential_light":  [("planner", "Planner"), ("critic", "Critic"), ("solver", "Solver")],
    "sequential_scaled": [("planner", "Planner"), ("critic", "Critic"), ("solver", "Solver")],
    "mixture":           [("math", "Math"), ("code", "Code"), ("science", "Science"), ("summarizer", "Summarizer")],
    "distillation":      [("expert", "Expert"), ("learner", "Learner")],
    "deliberation":      [("reflector", "Reflector"), ("toolcaller", "Toolcaller")],
}

# Approximate per-agent VRAM at bfloat16 (GB) — used for pre-check warnings
_AGENT_VRAM_GB: Dict[str, Dict[str, float]] = {
    "sequential_light":  {"planner": 1.5, "critic": 1.0, "solver": 1.8},
    "sequential_scaled": {"planner": 3.5, "critic": 2.5, "solver": 3.5},
    "mixture":           {"math": 1.5, "code": 2.5, "science": 5.5, "summarizer": 1.5},
    "distillation":      {"expert": 8.0, "learner": 3.5},
    "deliberation":      {"reflector": 3.5, "toolcaller": 3.5},
}

# Approximate total VRAM per style (GB) — shown in Chat tab
_STYLE_VRAM_GB: Dict[str, float] = {
    "sequential_light": 5.0,
    "sequential_scaled": 12.0,
    "mixture": 15.0,
    "distillation": 18.0,
    "deliberation": 12.0,
}


def _available_devices() -> List[str]:
    devs: List[str] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devs.append(f"cuda:{i}")
    devs.append("cpu")
    return devs


def _vram_info() -> Dict[str, Tuple[float, float]]:
    """Return {device_str: (free_gb, total_gb)} for every CUDA device."""
    result: Dict[str, Tuple[float, float]] = {}
    if not torch.cuda.is_available():
        return result
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            result[f"cuda:{i}"] = (free / 1024 ** 3, total / 1024 ** 3)
        except Exception:
            pass
    return result


def _vram_status_md(device: str, style: str) -> str:
    """Return a one-line markdown VRAM status for the given device/style pair."""
    if device == "cpu":
        return "<small>CPU mode — no VRAM limit.</small>"
    info = _vram_info()
    if device not in info:
        return "<small>VRAM info unavailable.</small>"
    free, total = info[device]
    needed = _STYLE_VRAM_GB.get(style, 0.0)
    pct_used = (total - free) / total * 100 if total > 0 else 0
    icon = "🔴" if free < needed else "🟡" if free < needed * 1.25 else "🟢"
    line = f"{icon} **{device}**: {free:.1f} GB free / {total:.1f} GB &nbsp;({pct_used:.0f}% used)"
    if needed > 0:
        fit = "✅ fits" if free >= needed else f"⚠️ **may OOM** — {needed:.0f} GB needed, only {free:.1f} GB free"
        line += f"  \n<small>Style `{style}` needs ~{needed:.0f} GB — {fit}</small>"
    return line


# ── Global log buffer ──────────────────────────────────────────────────────────

_LOG_BUFFER: List[str] = []
_LOG_LOCK = threading.Lock()
_LOG_MAX_CHARS = 300_000  # ~300 KB rolling window


def _log_write(text: str) -> None:
    if not text:
        return
    with _LOG_LOCK:
        _LOG_BUFFER.append(text)
        total = sum(len(s) for s in _LOG_BUFFER)
        while total > _LOG_MAX_CHARS and len(_LOG_BUFFER) > 1:
            total -= len(_LOG_BUFFER.pop(0))


def _log_get() -> str:
    with _LOG_LOCK:
        return "".join(_LOG_BUFFER)


def _log_clear() -> str:
    with _LOG_LOCK:
        _LOG_BUFFER.clear()
    return ""


def _analyze_log(log_text: str) -> str:
    import re as _re
    if not log_text.strip():
        return "*No log content — run an inference or batch evaluation first.*"

    issues: List[str] = []
    warnings_found: List[str] = []
    successes: List[str] = []

    # OOM
    if _re.search(r"OutOfMemoryError|CUDA out of memory", log_text):
        devs = _re.findall(r"(cuda:\d+|GPU \d+)", log_text)
        label = ", ".join(sorted(set(devs))) if devs else "unknown device"
        issues.append(f"GPU OOM on **{label}** — reduce batch size or use a style with lower VRAM requirements")

    # Python errors (first 3 unique)
    errs = _re.findall(r"((?:RuntimeError|ValueError|KeyError|TypeError|AttributeError|ImportError): .+)", log_text)
    seen: set = set()
    for e in errs:
        short = e[:140]
        if short not in seen:
            seen.add(short)
            issues.append(f"`{short}`")
        if len(seen) >= 3:
            break

    # Download / setup failures
    if _re.search(r"❌ Download failed|❌ Failed `|❌ Setup error", log_text):
        issues.append("Model download or setup failed — check network and Model Manager")

    # Incomplete model cache
    if _re.search(r"⚠️ incomplete", log_text):
        warnings_found.append("Incomplete model cache detected — use 🔄 Update in Model Manager")

    # User stop
    if _re.search(r"⏹ Stopped|⏹ Batch stopped", log_text):
        warnings_found.append("Batch run was stopped by user")

    # Accuracy results
    accs = _re.findall(r"accuracy=([0-9]+(?:\.[0-9]+)?)%", log_text)
    for a in accs:
        successes.append(f"Accuracy result: **{a}%**")

    # Multi-GPU activation lines
    mgpu = _re.findall(r"\[multi-gpu\] (.+)", log_text)
    for line in mgpu[:3]:
        successes.append(f"Multi-GPU active: `{line.strip()}`")

    # Model cache events
    loads = len(_re.findall(r"\[serve\] loading", log_text))
    hits  = len(_re.findall(r"\[serve\] cache hit", log_text))
    if loads or hits:
        successes.append(f"Model loads: **{loads}**, cache hits: **{hits}**")

    # Batch completion
    if _re.search(r"✅ Batch complete", log_text):
        successes.append("Batch evaluation completed successfully")

    if not issues and not warnings_found and not successes:
        return "*No notable events found.*"

    parts: List[str] = []
    if issues:
        parts.append("### ❌ Issues\n" + "\n".join(f"- {i}" for i in issues))
    if warnings_found:
        parts.append("### ⚠️ Warnings\n" + "\n".join(f"- {w}" for w in warnings_found))
    if successes:
        parts.append("### ✅ Events\n" + "\n".join(f"- {s}" for s in successes))
    return "\n\n".join(parts)



def _mg_update_style(style: str, enabled: bool):
    """Return gr.update() tuples for the 4 agent rows when the style changes."""
    roles = _STYLE_AGENT_ROLES.get(style, [])
    out = []
    for i in range(4):
        visible = enabled and i < len(roles)
        label = f"**{roles[i][1]}** (`{roles[i][0]}`)" if i < len(roles) else ""
        out += [gr.update(visible=visible), gr.update(value=label)]
    return out  # 8 items: row_vis, lbl_val × 4


def _mg_toggle_enabled(enabled: bool, style: str):
    """Show/hide agent rows + apply button when the enable checkbox changes."""
    roles = _STYLE_AGENT_ROLES.get(style, [])
    row_updates = [gr.update(visible=enabled and i < len(roles)) for i in range(4)]
    return [gr.update(visible=enabled)] + row_updates  # apply_btn + 4 rows


def _mg_apply(style: str, enabled: bool, dev0: str, dev1: str, dev2: str, dev3: str, state: dict):
    """Store the per-agent device assignment for *style* in the shared state."""
    new_state = dict(state or {})
    if not enabled:
        new_state.pop(style, None)
        return new_state, "*Multi-GPU disabled — agents use the global Device setting.*"
    roles = _STYLE_AGENT_ROLES.get(style, [])
    devs = [dev0, dev1, dev2, dev3]
    device_map = {role: devs[i] for i, (role, _) in enumerate(roles)}
    new_state[style] = device_map
    parts = [f"{lbl}→{devs[i]}" for i, (_, lbl) in enumerate(roles)]
    status_lines = [f"✅ **{style}** multi-GPU active: {', '.join(parts)}", ""]

    # Per-agent VRAM check
    vram = _vram_info()
    agent_vrams = _AGENT_VRAM_GB.get(style, {})
    for i, (role, lbl) in enumerate(roles):
        dev = devs[i]
        needed = agent_vrams.get(role, 0.0)
        if dev in vram:
            free, total = vram[dev]
            icon = "🟢" if free >= needed * 1.25 else "🟡" if free >= needed else "🔴"
            fit = "OK" if free >= needed else f"⚠️ **may OOM** ({needed:.1f} GB needed, {free:.1f} GB free)"
            status_lines.append(f"{icon} **{lbl}** on `{dev}`: ~{needed:.1f} GB — {fit}")
        else:
            status_lines.append(f"ℹ️ **{lbl}** on `{dev}`: ~{needed:.1f} GB needed (VRAM info unavailable)")

    return new_state, "\n".join(status_lines)


# ── Gradio layout ─────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    device_opts = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

    _vram_note = (
        "**Approx. VRAM**\n"
        "- `sequential_light` ≈ 5 GB\n"
        "- `sequential_scaled` ≈ 12 GB\n"
        "- `mixture` ≈ 15 GB\n"
        "- `distillation` ≈ 18 GB\n"
        "- `deliberation` ≈ 12 GB"
    )

    all_devs = _available_devices()

    with gr.Blocks(title=f"HOUSE — RecursiveMAS v{_VERSION}") as demo:
        gr.Markdown(
            f"# 🏥 HOUSE &nbsp;<sup style='font-size:0.5em;color:#888'>v{_VERSION}</sup>\n"
            "### *Multi-agent diagnostic reasoning via latent-space recursion*\n"
            "> Inspired by Dr. Gregory House — three specialist agents debate, refine, and solve.  \n"
            "> Models load into VRAM on first request and stay warm for subsequent ones."
        )

        # Shared state: maps style_name → {role: device_str}
        device_map_state = gr.State({})

        with gr.Tabs():

            # ── Tab 1: Chat ───────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        gr.Markdown("### Settings")
                        style_dd = gr.Dropdown(
                            choices=list(STYLE_SPECS.keys()),
                            value="sequential_light",
                            label="Collaboration style",
                        )
                        style_info = gr.Markdown(
                            _STYLE_DESCRIPTIONS["sequential_light"],
                        )
                        gr.Markdown(
                            "<small>ℹ️ See the <b>📐 Architectures</b> tab for pipeline diagrams.</small>"
                        )
                        domain_dd = gr.Dropdown(
                            choices=list(DOMAIN_SYSTEM_PROMPTS.keys()),
                            value="general",
                            label="Reasoning domain",
                        )
                        gr.Markdown(
                            "> ℹ️ **First use:** model weights are downloaded from HuggingFace "
                            "and cached locally. This may take several minutes. "
                            "Subsequent runs load from cache instantly."
                        )
                        rounds_sl = gr.Slider(1, 5, value=3, step=1, label="Recursive rounds")
                        latent_sl = gr.Slider(8, 64, value=32, step=8, label="Latent steps")
                        device_dd = gr.Dropdown(choices=device_opts, value=device_opts[0], label="Device")
                        vram_status_md = gr.Markdown(
                            _vram_status_md(device_opts[0], "sequential_light"),
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            temperature_sl = gr.Slider(
                                0.0, 1.0, value=0.6, step=0.05,
                                label="Temperature",
                                info="Higher = more creative, lower = more deterministic",
                            )
                            top_p_sl = gr.Slider(
                                0.0, 1.0, value=0.95, step=0.05,
                                label="Top-p (nucleus sampling)",
                                info="Cumulative probability threshold for token selection",
                            )
                            seed_num = gr.Number(
                                value=42, precision=0,
                                label="Seed",
                                info="Fixed seed for reproducible outputs (integer)",
                            )

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=520, label="", show_label=False)
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Ask a math, science, or reasoning question…",
                                label="", lines=2, scale=5, show_label=False,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        gr.Button("Clear").click(lambda: ([], []), outputs=[chatbot, gr.State([])])

                state = gr.State([])
                for trigger in (send_btn.click, msg.submit):
                    trigger(
                        respond,
                        inputs=[
                            msg, state, style_dd, domain_dd,
                            rounds_sl, latent_sl, device_dd,
                            temperature_sl, top_p_sl, seed_num,
                            device_map_state,
                        ],
                        outputs=[chatbot, state, msg],
                    )
                style_dd.change(
                    lambda s: _STYLE_DESCRIPTIONS.get(s, ""),
                    inputs=[style_dd],
                    outputs=[style_info],
                )
                style_dd.change(
                    _vram_status_md,
                    inputs=[device_dd, style_dd],
                    outputs=[vram_status_md],
                )
                device_dd.change(
                    _vram_status_md,
                    inputs=[device_dd, style_dd],
                    outputs=[vram_status_md],
                )

            # ── Tab 2: Batch Evaluation ───────────────────────────────────
            with gr.Tab("📊 Batch Evaluation"):
                gr.Markdown(
                    "### Run a full benchmark evaluation\n"
                    "The pipeline processes every question in the selected dataset and streams "
                    "live progress below. Results are saved to a JSONL file available for download "
                    "when the run completes."
                )
                with gr.Row():
                    # ── Batch settings ────────────────────────────────────
                    with gr.Column(scale=1, min_width=260):
                        b_style_dd = gr.Dropdown(
                            choices=list(STYLE_SPECS.keys()),
                            value="sequential_light",
                            label="Collaboration style",
                        )
                        b_dataset_dd = gr.Dropdown(
                            choices=_BATCH_DATASETS,
                            value="math500",
                            label="Dataset",
                        )
                        b_samples_num = gr.Number(
                            value=-1, precision=0,
                            label="N samples (−1 = full benchmark)",
                            info="Positive integer to evaluate a random subset",
                        )
                        b_device_dd = gr.Dropdown(choices=device_opts, value=device_opts[0], label="Device")
                        b_rounds_sl = gr.Slider(1, 5, value=3, step=1, label="Recursive rounds")
                        b_latent_sl = gr.Slider(8, 64, value=32, step=8, label="Latent steps")
                        gr.Markdown(_vram_note)
                        with gr.Accordion("Advanced settings", open=False):
                            b_temp_sl = gr.Slider(
                                0.0, 1.0, value=0.6, step=0.05,
                                label="Temperature",
                                info="Higher = more creative, lower = more deterministic",
                            )
                            b_topp_sl = gr.Slider(
                                0.0, 1.0, value=0.95, step=0.05,
                                label="Top-p (nucleus sampling)",
                                info="Cumulative probability threshold for token selection",
                            )
                            b_seed_num = gr.Number(
                                value=42, precision=0,
                                label="Seed",
                                info="Fixed seed for reproducible outputs",
                            )
                        with gr.Row():
                            run_btn = gr.Button("▶ Run Batch", variant="primary", scale=2)
                            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1,
                                                 interactive=False)

                    # ── Batch output ──────────────────────────────────────
                    with gr.Column(scale=3):
                        batch_log = gr.Code(
                            label="Progress log",
                            language=None,
                            lines=30,
                            interactive=False,
                        )
                        dl_file = gr.File(
                            label="⬇ Download results (JSONL)",
                            visible=False,
                            interactive=False,
                        )

                run_btn.click(
                    run_batch_eval,
                    inputs=[
                        b_style_dd, b_dataset_dd, b_samples_num, b_device_dd,
                        b_rounds_sl, b_latent_sl, b_temp_sl, b_topp_sl, b_seed_num,
                        device_map_state,
                    ],
                    outputs=[batch_log, dl_file, run_btn, stop_btn],
                )
                stop_btn.click(stop_batch_eval)

            # ── Tab 3: Model Manager ──────────────────────────────────────
            with gr.Tab("📦 Model Manager"):
                gr.Markdown(
                    "### Model catalog\n"
                    "Tick one or more models, choose an action, then click **Run**. "
                    "Multiple selections are processed one at a time."
                )

                with gr.Row():
                    mm_refresh_btn  = gr.Button("🔄 Refresh",      size="sm", variant="secondary")
                    mm_sel_all_btn  = gr.Button("☑ Select All",    size="sm")
                    mm_desel_all_btn = gr.Button("☐ Deselect All", size="sm")
                    mm_action_dd = gr.Dropdown(
                        choices=["⬇ Download", "🔄 Update", "🔍 Check", "🗑 Delete"],
                        value=None,
                        label="Action",
                        scale=2,
                        min_width=180,
                    )
                    mm_run_btn = gr.Button("▶ Run", variant="primary", scale=1)

                mm_catalog_df = gr.Dataframe(
                    headers=["", "Style", "Role", "Repository", "Status", "Size"],
                    value=_build_model_catalog_data(),
                    datatype=["bool", "str", "str", "str", "str", "str"],
                    interactive=True,
                    wrap=True,
                    type="array",
                )

                mm_log = gr.Code(
                    label="Progress", language=None, lines=6, interactive=False
                )

                def _mm_toggle_all(data, checked: bool):
                    rows = data if data else _build_model_catalog_data()
                    return [[checked] + r[1:] for r in rows]

                mm_sel_all_btn.click(
                    lambda d: _mm_toggle_all(d, True),
                    inputs=[mm_catalog_df], outputs=[mm_catalog_df],
                )
                mm_desel_all_btn.click(
                    lambda d: _mm_toggle_all(d, False),
                    inputs=[mm_catalog_df], outputs=[mm_catalog_df],
                )
                mm_refresh_btn.click(
                    lambda: _build_model_catalog_data(),
                    outputs=[mm_catalog_df],
                )
                mm_run_btn.click(
                    _mm_run_action,
                    inputs=[mm_action_dd, mm_catalog_df],
                    outputs=[mm_log, mm_catalog_df],
                )

            # ── Tab 4: Architectures & Multi-GPU ─────────────────────────
            with gr.Tab("📐 Architectures"):
                gr.Markdown(
                    "### Collaboration style — pipeline diagrams\n"
                    "Each style defines a different multi-agent topology. "
                    "Agents communicate exclusively through **latent-space tensors** "
                    "via trained **RecursiveLink (RL)** adapters — no text is exchanged between agents within a round."
                )
                gr.HTML(_ARCH_HTML)

                gr.HTML("<hr style='margin:28px 0 20px;border:none;border-top:2px solid #e0e0e0'>")

                with gr.Accordion("⚙️ Multi-GPU Configuration (Experimental)", open=False):
                    gr.Markdown(
                        "Assign each agent in a collaboration style to a different GPU.  \n"
                        "Latent tensors move between devices automatically (`.to(device)`) — "
                        "the RecursiveLink adapters always run on the **source agent's device**.  \n"
                        "In Mixture style the three specialists can run **in parallel** on separate GPUs.  \n\n"
                        "⚠️ *Requires at least 2 CUDA devices. Default (single-GPU) behaviour is preserved when disabled.*"
                    )

                    with gr.Row():
                        mg_style_dd = gr.Dropdown(
                            choices=list(STYLE_SPECS.keys()),
                            value="sequential_light",
                            label="Style to configure",
                            scale=2,
                        )
                        mg_enabled_cb = gr.Checkbox(
                            value=False,
                            label="Enable multi-GPU for this style",
                            scale=1,
                        )

                    # 4 agent rows (max = mixture with 4 agents)
                    _init_roles = _STYLE_AGENT_ROLES["sequential_light"]
                    mg_row_comps: List[Tuple] = []
                    for _i in range(4):
                        _visible = False  # hidden until checkbox enabled
                        _label = f"**{_init_roles[_i][1]}** (`{_init_roles[_i][0]}`)" if _i < len(_init_roles) else ""
                        with gr.Row(visible=_visible) as _mg_row:
                            _mg_lbl = gr.Markdown(_label)
                            _mg_dev = gr.Dropdown(
                                choices=all_devs,
                                value=all_devs[0] if all_devs else "cpu",
                                label=f"Agent {_i + 1} device",
                                scale=2,
                            )
                        mg_row_comps.append((_mg_row, _mg_lbl, _mg_dev))

                    mg_apply_btn = gr.Button("💾 Apply Configuration", variant="primary", visible=False)
                    mg_status = gr.Markdown(
                        "*Multi-GPU disabled — all agents use the global Device setting.*"
                    )

                    # ── events ───────────────────────────────────────────────
                    _mg_row_outputs = []
                    for _mg_row, _mg_lbl, _ in mg_row_comps:
                        _mg_row_outputs += [_mg_row, _mg_lbl]

                    mg_style_dd.change(
                        _mg_update_style,
                        inputs=[mg_style_dd, mg_enabled_cb],
                        outputs=_mg_row_outputs,
                    )

                    mg_enabled_cb.change(
                        _mg_toggle_enabled,
                        inputs=[mg_enabled_cb, mg_style_dd],
                        outputs=[mg_apply_btn] + [row for row, _, _ in mg_row_comps],
                    )

                    mg_apply_btn.click(
                        _mg_apply,
                        inputs=[
                            mg_style_dd, mg_enabled_cb,
                            mg_row_comps[0][2], mg_row_comps[1][2],
                            mg_row_comps[2][2], mg_row_comps[3][2],
                            device_map_state,
                        ],
                        outputs=[device_map_state, mg_status],
                    )

            # ── Tab 5: Logs ───────────────────────────────────────────────
            with gr.Tab("📋 Logs"):
                gr.Markdown(
                    "### Inference & system log\n"
                    "Captures all output from chat inference, batch evaluation, and model downloads. "
                    "Click **🔍 Analyze** to scan for errors, OOM events, and accuracy results."
                )
                with gr.Row():
                    log_refresh_btn  = gr.Button("🔄 Refresh", size="sm", variant="secondary")
                    log_clear_btn    = gr.Button("🗑 Clear",   size="sm")
                    log_analyze_btn  = gr.Button("🔍 Analyze", size="sm", variant="primary")

                log_display = gr.Code(
                    label="Log output",
                    language=None,
                    lines=28,
                    interactive=False,
                    value="",
                )
                log_analysis_md = gr.Markdown("*Click 🔍 Analyze to scan for issues and events.*")

                log_refresh_btn.click(
                    lambda: _log_get(),
                    outputs=[log_display],
                )
                log_clear_btn.click(
                    lambda: (_log_clear(), "*Log cleared.*"),
                    outputs=[log_display, log_analysis_md],
                )
                log_analyze_btn.click(
                    lambda: (_log_get(), _analyze_log(_log_get())),
                    outputs=[log_display, log_analysis_md],
                )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser(description="RecursiveMAS web UI")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=7860, help="Bind port")
    p.add_argument("--share", action="store_true", help="Create a public Gradio tunnel")
    cfg = p.parse_args()

    build_ui().launch(server_name=cfg.host, server_port=cfg.port, share=cfg.share, theme=gr.themes.Soft())
