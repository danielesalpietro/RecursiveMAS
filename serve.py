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
        )
        cli_args += ["--result_jsonl", result_jsonl, "--num_samples", "-1"]

        # Run with stdout captured
        captured = io.StringIO()
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

    try:
        t_start = datetime.now()
        stdout, parsed = _run_single_question(
            style, message, device, num_rounds, latent_steps, domain,
            temperature=temperature, top_p=top_p, seed=seed,
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

    result_jsonl = tempfile.mktemp(suffix=".jsonl", prefix="house_batch_")
    log_q: "_queue.Queue[Optional[str]]" = _queue.Queue()

    thread = threading.Thread(
        target=_batch_worker,
        args=(style, dataset, num_samples, device, rounds, latent_steps,
              temperature, top_p, seed, result_jsonl, log_q),
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

    with gr.Blocks(title=f"HOUSE — RecursiveMAS v{_VERSION}") as demo:
        gr.Markdown(
            f"# 🏥 HOUSE &nbsp;<sup style='font-size:0.5em;color:#888'>v{_VERSION}</sup>\n"
            "### *Multi-agent diagnostic reasoning via latent-space recursion*\n"
            "> Inspired by Dr. Gregory House — three specialist agents debate, refine, and solve.  \n"
            "> Models load into VRAM on first request and stay warm for subsequent ones."
        )

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
                        gr.Markdown(_vram_note)
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
                        ],
                        outputs=[chatbot, state, msg],
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
