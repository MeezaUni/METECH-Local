"""Figures and summary tables for the project report.

    cd backend
    uv sync --group report
    uv run python ../ml/make_figures.py      # -> ml/reports/figures/*.png, ml/reports/summary.md

Reads the evaluation reports written by evaluate.py and the training history saved with each
model (copied into ml/reports/ so the figures can be rebuilt without the model files).
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
# KMRL_FIG_MODE=report: white background, no titles inside the images (the report gives numbered captions).
REPORT_MODE = os.environ.get("KMRL_FIG_MODE") == "report"
FIGS = REPORTS / "figures" / "report" if REPORT_MODE else REPORTS / "figures"
MODELS = ROOT.parent / "backend" / "models"

# Validated categorical palette (dataviz reference slots 1-3; all-pairs CVD-safe), fixed per entity.
ROUND2, ROUND1, RULES = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b7b0"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e5df", "#ffffff" if REPORT_MODE else "#fcfcfb"

# Gemini scores come from the cloud edition's evaluation on the same test set (METECH-Cloud/eval).
GEMINI = {"logbook": 1.0, "questions": 1.0}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def load(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def history(round_name: str, model_dir: str) -> list[dict]:
    """Training history, cached in reports/ so it survives without the model folder."""
    cached = REPORTS / f"training_history_{round_name}.json"
    meta = MODELS / model_dir / "kmrl_model.json"
    if meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        cached.write_text(json.dumps({k: v for k, v in info.items()}, indent=2), encoding="utf-8")
    return json.loads(cached.read_text(encoding="utf-8"))["history"] if cached.exists() else []


def title(ax, text: str, **kw) -> None:
    if not REPORT_MODE:
        ax.set_title(text, **kw)


def grid(ax, axis: str) -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- 1. loss curves
def fig_loss() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name, folder, color, label in (("round1", "kmrl-logreader-round1", ROUND1, "Round 1 (8,000 examples)"),
                                       ("round2", "kmrl-logreader", ROUND2, "Round 2 (9,199 examples)")):
        h = history(name, folder)
        train = [(x["step"], x["loss"]) for x in h if "loss" in x]
        val = [(x["step"], x["val_loss"]) for x in h if x.get("val_loss") is not None]
        if not train:
            continue
        xs, ys = zip(*train)
        # The final validation loss rides in the legend: labels at the curve ends collide (both end near 0.002).
        val_note = f", validation {val[-1][1]:.4f} (dot)" if val else ""
        ax.plot(xs, ys, color=color, linewidth=2, label=f"{label}{val_note}")
        for vx, vy in val:
            ax.plot([vx], [vy], marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
    ax.set_yscale("log")
    ax.set_xlabel("Optimiser step (effective batch 16)")
    ax.set_ylabel("Cross-entropy loss (log scale)")
    title(ax, "Training loss: both rounds converge within one epoch")
    grid(ax, "y")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_training_loss.png", dpi=200)
    plt.close(fig)


# -------------------------------------------------------------- 2. comparison
ENGINES = [  # (report file, label, is_ours)
    ("eval_rules.json", "Rule engine (regex)", False),
    ("eval_base.json", "Qwen2.5-0.5B, no fine-tuning (few-shot)", False),
    ("eval_ollama.json", "qwen3.5 9.7B, no fine-tuning", False),
    ("eval_model_round1.json", "kmrl-logreader round 1", False),
    ("eval_model_round2.json", "kmrl-logreader round 2 (ours)", True),
]


def results_rows() -> list[dict]:
    rows = []
    for file, label, ours in ENGINES:
        r = load(file)
        if not r or "realworld" not in r:
            continue
        rw = r["realworld"]
        rows.append({"label": label, "ours": ours, "logbook": rw["parse_exact_rate"], "questions": rw["ask_exact_rate"],
                     "precision": rw["fact_precision"], "recall": rw["fact_recall"],
                     "synthetic_logbook": r.get("synthetic", {}).get("parse_exact_rate")})
    rows.append({"label": "Gemini (cloud API, reference)", "ours": False, "logbook": GEMINI["logbook"],
                 "questions": GEMINI["questions"], "precision": 1.0, "recall": 1.0, "synthetic_logbook": None})
    return rows


def fig_comparison(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    labels = [r["label"] for r in rows]
    for ax, key, title in ((axes[0], "logbook", "Logbook messages fully correct"),
                           (axes[1], "questions", "Questions understood")):
        vals = [r[key] * 100 for r in rows]
        colors = [ROUND2 if r["ours"] else NEUTRAL for r in rows]
        ys = range(len(rows))
        ax.barh(ys, vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + 1.2, y, f"{v:.1f}%", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title)
        grid(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if not REPORT_MODE:
        fig.text(0.01, 0.015, "Hand-written real-world test set (43 logbook messages, 25 questions)", ha="left", va="bottom", color=INK_2, fontsize=9.5)
    fig.savefig(FIGS / "fig2_results_comparison.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------- 3. per-action recall
ACTION_LABELS = {
    "renew_certificate": "Renew certificate", "close_job_card": "Close job card", "open_job_card": "Open job card",
    "flag_cleaning": "Flag cleaning", "mark_cleaned": "Mark cleaned", "complete_maintenance": "Inspection done",
    "update_mileage": "Update mileage",
}


def per_action(report: dict) -> dict[str, float]:
    """Recall per action over both test sets combined (synthetic gives volume, real-world gives realism)."""
    totals: Counter = Counter()
    found: Counter = Counter()
    for split in ("synthetic", "realworld"):
        for a, v in report.get(split, {}).get("per_action", {}).items():
            totals[a] += v["expected"]
            found[a] += v["found"]
    return {a: found[a] / totals[a] for a in totals if totals[a]}


def fig_per_action() -> dict:
    series = [(load("eval_model_round2.json"), "Round 2 (ours)", ROUND2),
              (load("eval_model_round1.json"), "Round 1", ROUND1),
              (load("eval_rules.json"), "Rule engine", RULES)]
    series = [(per_action(r), n, c) for r, n, c in series if r]
    if not series:
        return {}
    actions = [a for a in ACTION_LABELS if a in series[0][0]]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    h = 0.26
    for i, (vals, name, color) in enumerate(series):
        ys = [k + (i - 1) * h for k in range(len(actions))]
        ax.barh(ys, [vals.get(a, 0) * 100 for a in actions], height=h, color=color, edgecolor=SURFACE, linewidth=2, label=name)
    for k, a in enumerate(actions):   # label only our model's bars, to keep the chart quiet
        v = series[0][0].get(a, 0) * 100
        ax.text(v + 1, k - h, (f"{v:.1f}%" if v < 99.95 else "100%"), va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(actions)), [ACTION_LABELS[a] for a in actions])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Recall: share of expected updates extracted (synthetic + real-world test sets)")
    title(ax, "Where the gains come from, by update type", pad=30)
    if REPORT_MODE:
        ax.set_title(" ", pad=22)   # room for the legend above the bars
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)   # above the plot, clear of the bars
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_per_action_recall.png", dpi=200)
    plt.close(fig)
    return {name: vals for vals, name, _ in series}


# --------------------------------------------------------- 4. dataset composition
def fig_dataset() -> dict:
    counts: Counter = Counter()
    teacher = 0
    path = ROOT / "data" / "train.jsonl"
    if not path.exists():
        return {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        target = json.loads(r["target"])
        if r["task"] == "ask":
            counts["Question to query"] += 1
        elif not target["updates"]:
            counts["No update (chatter)"] += 1
        else:
            for u in target["updates"]:
                counts[ACTION_LABELS[u["action"]]] += 1
    teacher_file = ROOT / "data" / "paraphrased.jsonl"
    if teacher_file.exists():
        teacher = sum(1 for _ in open(teacher_file, encoding="utf-8"))
    items = counts.most_common()
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ys = range(len(items))
    ax.barh(ys, [v for _, v in items], color=ROUND2, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, (_, v) in zip(ys, items):
        ax.text(v + 40, y, f"{v:,}", va="center", color=INK, fontsize=9)
    ax.set_yticks(ys, [k for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("Labelled items in the round-2 training set")
    title(ax, "Training data: what the model learned from")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(v for _, v in items) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_training_data.png", dpi=200)
    plt.close(fig)
    return {"items": dict(items), "teacher_rewrites": teacher}


# ------------------------------------------------------------------- summary
def summary(rows: list[dict], actions: dict, data: dict) -> None:
    pct = lambda v: "-" if v is None else f"{v * 100:.1f}%"   # noqa: E731
    lines = ["# Results summary (generated by make_figures.py)", "",
             "## Real-world test set (43 logbook messages, 25 questions)", "",
             "| Engine | Logbook fully correct | Fact precision | Fact recall | Questions | Synthetic logbook |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {pct(r['logbook'])} | {pct(r['precision'])} | {pct(r['recall'])} | "
                     f"{pct(r['questions'])} | {pct(r['synthetic_logbook'])} |")
    if actions:
        names = list(actions)
        lines += ["", "## Recall by update type (both test sets)", "", "| Update type | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for a, label in ACTION_LABELS.items():
            if a in actions[names[0]]:
                lines.append(f"| {label} | " + " | ".join(pct(actions[n].get(a)) for n in names) + " |")
    if data:
        lines += ["", "## Training data (round 2)", "", "| Item | Count |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in data["items"].items()]
        lines.append(f"| Teacher rewrites produced (90% train / 10% validation) | {data['teacher_rewrites']:,} |")
    (REPORTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_loss()
    rows = results_rows()
    fig_comparison(rows)
    actions = fig_per_action()
    data = fig_dataset()
    summary(rows, actions, data)
    print("wrote", ", ".join(sorted(p.name for p in FIGS.glob("*.png"))), "and summary.md")


if __name__ == "__main__":
    main()
