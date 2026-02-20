"""Local HTML viewer for ARC-AGI-3 benchmark results."""
from __future__ import annotations

import argparse
import io
import json
import posixpath
import time
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from PIL import Image

from arcagi3.utils.image import grid_to_image

_JSON_CACHE_LOCK = RLock()
_JSON_CACHE: Dict[Path, tuple[int, int, Any]] = {}

_HTML_CACHE_LOCK = RLock()
_HTML_CACHE: Optional[tuple[int, int, str]] = None

_RUN_ROWS_CACHE_LOCK = RLock()
_RUN_ROWS_CACHE: Dict[Path, tuple[float, List[Dict[str, Any]]]] = {}
_RUN_ROWS_CACHE_TTL_SECONDS = 1.5


def _file_signature(path: Path) -> Optional[tuple[int, int]]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return stat.st_mtime_ns, stat.st_size


def _safe_run_dir(results_root: Path, run_id: str) -> Optional[Path]:
    run_id = run_id.strip()
    if not run_id:
        return None
    candidate = (results_root / run_id).resolve()
    try:
        candidate.relative_to(results_root.resolve())
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_dir():
        return None
    return candidate


def _load_json(path: Path) -> Optional[Any]:
    signature = _file_signature(path)
    if signature is None:
        return None

    cache_key = path.resolve()
    with _JSON_CACHE_LOCK:
        cached = _JSON_CACHE.get(cache_key)
    if cached and cached[0] == signature[0] and cached[1] == signature[1]:
        return cached[2]

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[cache_key] = (signature[0], signature[1], parsed)
    return parsed


def _list_runs(results_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not results_root.exists():
        return runs

    for item in results_root.iterdir():
        if not item.is_dir():
            continue
        summary = _load_json(item / "summary.json") or {}
        runs.append(
            {
                "run_id": item.name,
                "execution_start": summary.get("execution_start"),
                "execution_end": summary.get("execution_end"),
                "total_games": summary.get("total_games"),
                "total_executions": summary.get("total_executions"),
                "models_tested": summary.get("models_tested", []),
                "has_summary": bool(summary),
            }
        )

    runs.sort(key=lambda x: x["run_id"], reverse=True)
    return runs


def _compute_fallback_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _cost(row: Dict[str, Any]) -> float:
        total_cost = row.get("total_cost")
        if isinstance(total_cost, dict):
            return float(total_cost.get("total_cost", 0.0) or 0.0)
        return float(total_cost or 0.0)

    def _tokens(row: Dict[str, Any]) -> int:
        usage = row.get("usage")
        if isinstance(usage, dict):
            return int(usage.get("total_tokens", 0) or 0)
        return int(row.get("total_tokens", 0) or 0)

    wins = sum(1 for r in results if r.get("final_state") == "WIN")
    game_overs = sum(1 for r in results if r.get("final_state") == "GAME_OVER")
    in_progress = sum(1 for r in results if r.get("final_state") == "IN_PROGRESS")

    costs = [_cost(r) for r in results]
    tokens = [_tokens(r) for r in results]
    scores = [float(r.get("final_score", 0) or 0) for r in results]
    actions = [int(r.get("actions_taken", 0) or 0) for r in results]
    durations = [float(r.get("duration_seconds", 0.0) or 0.0) for r in results]
    models = sorted({str(r.get("config", "unknown")) for r in results})

    return {
        "total_executions": len(results),
        "models_tested": models,
        "overall_stats": {
            "wins": wins,
            "game_overs": game_overs,
            "in_progress": in_progress,
            "total_cost": sum(costs),
            "total_tokens": sum(tokens),
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "avg_actions": (sum(actions) / len(actions)) if actions else 0.0,
            "total_duration_seconds": sum(durations),
        },
    }


def _read_results_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    run_key = run_dir.resolve()
    now = time.monotonic()
    with _RUN_ROWS_CACHE_LOCK:
        cached = _RUN_ROWS_CACHE.get(run_key)
    if cached and (now - cached[0]) <= _RUN_ROWS_CACHE_TTL_SECONDS:
        return cached[1]

    rows_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for game_dir in run_dir.iterdir():
        if not game_dir.is_dir() or game_dir.name.startswith(".") or game_dir.name.startswith("_"):
            continue
        for result_file in game_dir.glob("*.json"):
            data = _load_json(result_file)
            if not isinstance(data, dict):
                continue
            timestamp = data.get("timestamp") or data.get("updated_at")
            is_live = result_file.name.endswith("_live.json")
            row = {
                "file": str(result_file.relative_to(run_dir)).replace("\\", "/"),
                "game_id": data.get("game_id", game_dir.name),
                "config": data.get("config", "unknown"),
                "final_state": data.get("final_state", "unknown"),
                "final_score": data.get("final_score", 0),
                "actions_taken": data.get("actions_taken", 0),
                "duration_seconds": data.get("duration_seconds", 0.0),
                "total_cost": (data.get("total_cost") or {}).get("total_cost", 0.0),
                "total_tokens": (data.get("usage") or {}).get("total_tokens", 0),
                "timestamp": timestamp,
                "is_live": is_live,
            }
            key = (str(row["game_id"]), str(row["config"]))
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = row
                continue

            existing_is_live = bool(existing.get("is_live"))
            existing_ts = str(existing.get("timestamp") or "")
            current_ts = str(row.get("timestamp") or "")

            # Prefer final file over live file for same game/config; otherwise prefer latest timestamp.
            if existing_is_live and not is_live:
                rows_by_key[key] = row
            elif existing_is_live == is_live and current_ts >= existing_ts:
                rows_by_key[key] = row

    rows = list(rows_by_key.values())
    rows.sort(key=lambda x: str(x.get("timestamp") or ""), reverse=True)
    with _RUN_ROWS_CACHE_LOCK:
        _RUN_ROWS_CACHE[run_key] = (time.monotonic(), rows)
    return rows


def _safe_result_file(run_dir: Path, result_file: str) -> Optional[Path]:
    result_file = result_file.strip().replace("\\", "/")
    if not result_file:
        return None
    candidate = (run_dir / result_file).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file() or candidate.suffix.lower() != ".json":
        return None
    return candidate


def _augment_result_with_checkpoint_frames(
    result: Dict[str, Any], checkpoint_roots: List[Path]
) -> Dict[str, Any]:
    """
    If result actions are missing optional diagnostics, try to hydrate them from
    .checkpoint/<card_id>/ files (action_history/model_completion/datastore_history).
    """
    actions = result.get("actions")
    card_id = result.get("card_id")
    if not isinstance(actions, list) or not card_id:
        return result

    missing_frame_data = True
    missing_memory_data = True
    missing_model_calls_data = True
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("frames_before") is not None or action.get("frames_after") is not None:
            missing_frame_data = False
        if action.get("memory_prompt") is not None:
            missing_memory_data = False
        if action.get("model_calls") is not None:
            missing_model_calls_data = False

    if not missing_frame_data and not missing_memory_data and not missing_model_calls_data:
        return result

    checkpoint_actions = None
    checkpoint_model_calls = None
    checkpoint_datastore_history: List[Dict[str, Any]] = []
    for checkpoint_root in checkpoint_roots:
        checkpoint_dir = checkpoint_root / str(card_id)
        history_path = checkpoint_dir / "action_history.json"
        maybe_actions = _load_json(history_path)
        if isinstance(maybe_actions, list):
            checkpoint_actions = maybe_actions
        model_calls_path = checkpoint_dir / "model_completion.json"
        maybe_model_calls = _load_json(model_calls_path)
        if isinstance(maybe_model_calls, list):
            checkpoint_model_calls = maybe_model_calls

        datastore_history_path = checkpoint_dir / "datastore_history.jsonl"
        if datastore_history_path.exists():
            try:
                with datastore_history_path.open("r", encoding="utf-8") as f:
                    parsed_lines: List[Dict[str, Any]] = []
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            parsed_lines.append(obj)
                    checkpoint_datastore_history = parsed_lines
            except Exception:
                checkpoint_datastore_history = []

        if (
            checkpoint_actions is not None
            or checkpoint_model_calls is not None
            or checkpoint_datastore_history
        ):
            break

    checkpoint_by_num: Dict[int, Dict[str, Any]] = {}
    if isinstance(checkpoint_actions, list):
        for entry in checkpoint_actions:
            if not isinstance(entry, dict):
                continue
            try:
                action_num = int(entry.get("action_num"))
            except Exception:
                continue
            checkpoint_by_num[action_num] = entry

    model_calls_by_action_num: Dict[int, List[Dict[str, Any]]] = {}
    if isinstance(checkpoint_model_calls, list):
        for call in checkpoint_model_calls:
            if not isinstance(call, dict):
                continue
            try:
                action_num = int(call.get("action_num"))
            except Exception:
                continue
            model_calls_by_action_num.setdefault(action_num, []).append(call)

    memory_by_action_num: Dict[int, str] = {}
    for entry in checkpoint_datastore_history:
        if not isinstance(entry, dict):
            continue
        try:
            action_num = int(entry.get("action_num"))
        except Exception:
            continue
        datastore = entry.get("datastore")
        if not isinstance(datastore, dict):
            continue
        memory_prompt = datastore.get("memory_prompt")
        if isinstance(memory_prompt, str):
            memory_by_action_num[action_num] = memory_prompt

    hydrated_actions: List[Dict[str, Any]] = []
    changed = False
    for action in actions:
        if not isinstance(action, dict):
            hydrated_actions.append(action)
            continue
        hydrated = dict(action)
        try:
            action_num = int(action.get("action_num"))
        except Exception:
            action_num = None
        if action_num is not None and action_num in checkpoint_by_num:
            cp = checkpoint_by_num[action_num]
            if hydrated.get("frames_before") is None and cp.get("frames_before") is not None:
                hydrated["frames_before"] = cp.get("frames_before")
                changed = True
            if hydrated.get("frames_after") is None and cp.get("frames_after") is not None:
                hydrated["frames_after"] = cp.get("frames_after")
                changed = True
        if action_num is not None and hydrated.get("memory_prompt") is None:
            if action_num in memory_by_action_num:
                hydrated["memory_prompt"] = memory_by_action_num[action_num]
                changed = True
        if action_num is not None and hydrated.get("model_calls") is None:
            if action_num in model_calls_by_action_num:
                hydrated["model_calls"] = model_calls_by_action_num[action_num]
                changed = True
        hydrated_actions.append(hydrated)

    if changed:
        out = dict(result)
        out["actions"] = hydrated_actions
        return out
    return result


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    text: str,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_viewer_html() -> str:
    here = Path(__file__).resolve().parent
    html_path = here / "static" / "results_viewer.html"
    signature = _file_signature(html_path)
    if signature is None:
        return html_path.read_text(encoding="utf-8")

    global _HTML_CACHE
    with _HTML_CACHE_LOCK:
        cached = _HTML_CACHE
    if cached and cached[0] == signature[0] and cached[1] == signature[1]:
        return cached[2]

    html = html_path.read_text(encoding="utf-8")
    with _HTML_CACHE_LOCK:
        _HTML_CACHE = (signature[0], signature[1], html)
    return html


def _compact_result_for_viewer(result: Dict[str, Any]) -> Dict[str, Any]:
    compact_actions: List[Dict[str, Any]] = []
    actions = result.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            compact_actions.append(
                {
                    "action_num": action.get("action_num"),
                    "action": action.get("action"),
                    "action_data": action.get("action_data"),
                    "reasoning": action.get("reasoning"),
                    "result_score": action.get("result_score"),
                    "result_state": action.get("result_state"),
                    "frames_before": action.get("frames_before"),
                    "frames_after": action.get("frames_after"),
                    "memory_prompt": action.get("memory_prompt"),
                    "model_calls": action.get("model_calls"),
                }
            )

    return {
        "game_id": result.get("game_id"),
        "config": result.get("config"),
        "final_score": result.get("final_score"),
        "final_state": result.get("final_state"),
        "actions_taken": result.get("actions_taken"),
        "duration_seconds": result.get("duration_seconds"),
        "timestamp": result.get("timestamp"),
        "scorecard_url": result.get("scorecard_url"),
        "card_id": result.get("card_id"),
        "error": result.get("error"),
        "actions": compact_actions,
    }


def _extract_grid(frames_value: Any) -> Optional[List[List[int]]]:
    if not isinstance(frames_value, list) or not frames_value:
        return None
    first = frames_value[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        return first
    if isinstance(first, list) and first and isinstance(first[0], int):
        return frames_value  # Already a 2D grid.
    return None


def _build_playback_grids(actions: List[Dict[str, Any]]) -> List[List[List[int]]]:
    grids: List[List[List[int]]] = []
    previous_grid: Optional[List[List[int]]] = None
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        before = _extract_grid(action.get("frames_before")) or previous_grid
        after = _extract_grid(action.get("frames_after")) or before or previous_grid

        if idx == 0 and before is not None:
            grids.append(before)
        if after is not None:
            grids.append(after)
            previous_grid = after
    return grids


def _gif_response(handler: BaseHTTPRequestHandler, grids: List[List[List[int]]], filename: str) -> None:
    if not grids:
        return _json_response(
            handler,
            {"error": "No frame grids available to generate GIF"},
            status=HTTPStatus.BAD_REQUEST,
        )

    frames: List[Image.Image] = []
    for grid in grids:
        frames.append(grid_to_image(grid).convert("P", palette=Image.ADAPTIVE))

    out = io.BytesIO()
    first, *rest = frames
    duration_ms = int(round(1000 / 3.0))  # 3 actions per second
    first.save(
        out,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    body = out.getvalue()

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in filename)
    if not safe_name.lower().endswith(".gif"):
        safe_name += ".gif"

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/gif")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(results_root: Path, checkpoint_root: Path):
    class ResultsViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = posixpath.normpath(parsed.path)

            if path in ("/", "/index.html"):
                return _text_response(self, _read_viewer_html(), content_type="text/html; charset=utf-8")

            if path == "/api/runs":
                return _json_response(self, {"runs": _list_runs(results_root)})

            if path.startswith("/api/run/"):
                parts = path.split("/")
                if len(parts) < 5:
                    return _json_response(self, {"error": "Bad request"}, status=HTTPStatus.BAD_REQUEST)

                run_id = urllib.parse.unquote(parts[3])
                action = parts[4]
                run_dir = _safe_run_dir(results_root, run_id)
                if run_dir is None:
                    return _json_response(self, {"error": "Run not found"}, status=HTTPStatus.NOT_FOUND)

                if action == "summary":
                    summary = _load_json(run_dir / "summary.json")
                    if summary is None:
                        summary = _compute_fallback_summary(_read_results_for_run(run_dir))
                    return _json_response(self, summary)

                if action == "results":
                    rows = _read_results_for_run(run_dir)
                    return _json_response(self, {"results": rows})

                if action == "result":
                    params = urllib.parse.parse_qs(parsed.query)
                    file_list = params.get("file") or []
                    if not file_list:
                        return _json_response(
                            self,
                            {"error": "Missing file query parameter"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                    result_path = _safe_result_file(run_dir, urllib.parse.unquote(file_list[0]))
                    if result_path is None:
                        return _json_response(
                            self, {"error": "Result file not found"}, status=HTTPStatus.NOT_FOUND
                        )
                    data = _load_json(result_path)
                    if not isinstance(data, dict):
                        return _json_response(
                            self, {"error": "Invalid JSON result file"}, status=HTTPStatus.BAD_REQUEST
                        )
                    data = _augment_result_with_checkpoint_frames(
                        data, [checkpoint_root, run_dir / ".checkpoint"]
                    )
                    if (params.get("view") or ["playback"])[0] != "full":
                        data = _compact_result_for_viewer(data)
                    return _json_response(self, {"result": data})

                if action == "gif":
                    params = urllib.parse.parse_qs(parsed.query)
                    file_list = params.get("file") or []
                    if not file_list:
                        return _json_response(
                            self,
                            {"error": "Missing file query parameter"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                    result_path = _safe_result_file(run_dir, urllib.parse.unquote(file_list[0]))
                    if result_path is None:
                        return _json_response(
                            self, {"error": "Result file not found"}, status=HTTPStatus.NOT_FOUND
                        )
                    data = _load_json(result_path)
                    if not isinstance(data, dict):
                        return _json_response(
                            self, {"error": "Invalid JSON result file"}, status=HTTPStatus.BAD_REQUEST
                        )
                    data = _augment_result_with_checkpoint_frames(
                        data, [checkpoint_root, run_dir / ".checkpoint"]
                    )
                    actions = data.get("actions")
                    if not isinstance(actions, list):
                        return _json_response(
                            self, {"error": "Result has no actions"}, status=HTTPStatus.BAD_REQUEST
                        )
                    grids = _build_playback_grids(actions)
                    game_id = str(data.get("game_id") or "game")
                    config = str(data.get("config") or "model")
                    return _gif_response(self, grids, f"{game_id}_{config}.gif")

            return _json_response(self, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Keep server output quiet and focused.
            return

    return ResultsViewerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HTML viewer for benchmark results.")
    parser.add_argument("--results-root", default="results", help="Path to results root directory.")
    parser.add_argument(
        "--checkpoint-root",
        default=".checkpoint",
        help="Path to checkpoint root directory (default: .checkpoint).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8020, help="Port to bind (default: 8020).")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root).resolve()

    handler_cls = make_handler(results_root, checkpoint_root)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    now = datetime.now().isoformat(timespec="seconds")
    print(f"[{now}] Results viewer running at http://{args.host}:{args.port}")
    print(f"[{now}] Results root: {results_root}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
