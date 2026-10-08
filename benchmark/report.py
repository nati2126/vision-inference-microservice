"""Render a benchmark result dict as markdown tables."""

from typing import Any


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _http(row: dict[str, Any], concurrency: int) -> dict[str, Any]:
    return next((r for r in row.get("http", []) if r["concurrency"] == concurrency), {})


def hardware_line(report: dict[str, Any]) -> str:
    hw = report["hardware"]
    gpu = hw["gpu"]["name"] + f" ({hw['gpu']['memory_gb']} GB)" if hw.get("gpu") else "no GPU"
    power = {True: "on AC power", False: "ON BATTERY", None: "power state unknown"}[
        hw.get("on_ac_power")
    ]
    return (
        f"{hw['cpu']} ({hw['cpu_cores_physical']} cores / {hw['cpu_cores_logical']} threads), "
        f"{hw['ram_gb']} GB RAM, {gpu}, {hw['os']}, Python {hw['python']}, {power}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["results"]
    ok = [r for r in rows if r.get("status") == "ok"]
    serving = [r for r in ok if not r["accuracy_only"]]
    settings = report["settings"]
    budget = settings["budget"]
    concurrencies = budget["http_concurrency"]

    lines = [
        f"# Benchmark results ({report['mode']} run)",
        "",
        f"- **Hardware:** {hardware_line(report)}",
        f"- **Generated:** {report['generated_at']}, commit `{report['hardware']['git_commit']}`,"
        f" took {report['duration_s']:.0f}s",
        f"- **Accuracy:** {settings['dataset']}, {budget['eval_images'] or 128} images,"
        f" conf {settings['eval_thresholds']['conf']}, NMS IoU {settings['eval_thresholds']['iou']}."
        " A relative comparison between backends, not an official COCO result.",
        f"- **Latency:** `predict()` end to end (preprocess + inference + postprocess), batch 1,"
        f" {budget['warmup_runs']} warm-up + {budget['timed_runs']} timed runs on one 640x480 image,"
        f" serving thresholds (conf {settings['serving_thresholds']['conf']}).",
        f"- **HTTP:** {budget['http_requests']} `POST /api/v1/detect` requests per concurrency level"
        " against uvicorn (1 worker); client and server share the machine.",
        "",
        "## Model",
        "",
        "| Backend | Device | Precision | mAP50 | mAP50-95 | Mean ms | p50 ms | p95 ms"
        " | img/s | Size MB | Peak RSS MB | Peak GPU MB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ok:
        lat = r.get("latency", {})
        name = r["backend"] + (" (ablation: naive INT8)" if r["accuracy_only"] else "")
        lines.append(
            f"| {name} | {r['device']} | {r['precision']} "
            f"| {_fmt(r['accuracy']['map50'], 3)} | {_fmt(r['accuracy']['map50_95'], 3)} "
            f"| {_fmt(lat.get('mean_ms'), 2)} | {_fmt(lat.get('p50_ms'), 2)} "
            f"| {_fmt(lat.get('p95_ms'), 2)} | {_fmt(lat.get('throughput_img_s'))} "
            f"| {_fmt(r['size_mb'], 1)} | {_fmt(r['peak_rss_mb'], 0)} "
            f"| {_fmt(r.get('peak_gpu_mem_mb'), 0)} |"
        )

    if any("http" in r for r in serving):
        header = "| Backend | Device | Precision |"
        rule = "|---|---|---|"
        for c in concurrencies:
            header += f" c={c} p50 ms | c={c} p95 ms | c={c} req/s |"
            rule += "---:|---:|---:|"
        lines += ["", "## HTTP", "", header, rule]
        for r in serving:
            cells = f"| {r['backend']} | {r['device']} | {r['precision']} |"
            for c in concurrencies:
                run = _http(r, c)
                errors = f" ({run['errors']} errors)" if run.get("errors") else ""
                cells += (
                    f" {_fmt(run.get('p50_ms'))} | {_fmt(run.get('p95_ms'))}"
                    f" | {_fmt(run.get('requests_per_s'))}{errors} |"
                )
            lines.append(cells)

    not_run = [r for r in rows if r.get("status") != "ok"]
    if not_run:
        lines += ["", "## Not run", ""]
        lines += [f"- `{r['name']}`: {r['status']}: {r['reason']}" for r in not_run]

    return "\n".join(lines) + "\n"
