"""
Reward head stats.

Reads reward_head/training_data/*.jsonl and writes:
- training_data/stats/reward_head_stats.csv
- training_data/stats/summary.txt
- training_data/graphs/loss_curve.svg
- training_data/graphs/predicted_vs_actual.svg

Each JSONL row is expected to have:
    step        int     global training step
    loss        float   MSE loss at this step
    predicted   float   reward_head output
    actual      float   ground-truth reward from reward.py
"""

from pathlib import Path
import argparse
import csv
import html
import json


def reward_head_stats(data_dir=None):

    # -------------------------------------------------------------------------
    # Step 1: read training JSONL rows
    # -------------------------------------------------------------------------

    data_dir   = Path(data_dir or Path(__file__).resolve().parent / "training_data")
    stats_dir  = data_dir / "stats"
    graphs_dir = data_dir / "graphs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_file"] = path.name
                row["_line"] = line_number
                rows.append(row)

    # -------------------------------------------------------------------------
    # Step 2: write flat CSV
    # -------------------------------------------------------------------------

    columns = [
        "source_file", "line", "global_index",
        "step", "loss", "predicted", "actual",
    ]

    csv_path = stats_dir / "reward_head_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({
                "source_file":  row.get("_file", ""),
                "line":         row.get("_line", ""),
                "global_index": index,
                "step":         row.get("step", ""),
                "loss":         row.get("loss", ""),
                "predicted":    row.get("predicted", ""),
                "actual":       row.get("actual", ""),
            })

    # -------------------------------------------------------------------------
    # Step 3: SVG helper
    # -------------------------------------------------------------------------

    def get_values(key):
        out = []
        for row in rows:
            v = row.get(key, None)
            if v is not None and v != "":
                out.append(float(v))
        return out

    def draw_svg(path, title, series):
        width, height = 960, 520
        left, right, top, bottom = 70, 30, 45, 70
        plot_w = width - left - right
        plot_h = height - top - bottom

        all_pts = []
        for _, key, _ in series:
            for i, row in enumerate(rows, start=1):
                v = row.get(key)
                if v is not None and v != "":
                    all_pts.append((i, float(v)))

        if not all_pts:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
                    f"<text x='24' y='32' font-family='Arial' font-size='22'>{html.escape(title)}</text>"
                    f"<text x='24' y='72' font-family='Arial' font-size='16'>No rows yet</text>"
                    "</svg>"
                )
            return

        min_x = min(p[0] for p in all_pts)
        max_x = max(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        max_y = max(p[1] for p in all_pts)
        if max_x == min_x: max_x = min_x + 1
        if max_y == min_y: max_y = min_y + 1.0

        def xp(x): return left + ((x - min_x) / (max_x - min_x)) * plot_w
        def yp(y): return top  + (1.0 - ((y - min_y) / (max_y - min_y))) * plot_h

        parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
            "<rect width='100%' height='100%' fill='white'/>",
            f"<text x='24' y='32' font-family='Arial' font-size='22' fill='#111'>{html.escape(title)}</text>",
            f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='#888'/>",
            f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='#888'/>",
            f"<text x='{left}' y='{height-25}' font-family='Arial' font-size='13' fill='#555'>samples: {len(rows)}</text>",
            f"<text x='{left}' y='{top-10}' font-family='Arial' font-size='12' fill='#555'>{max_y:.4f}</text>",
            f"<text x='{left}' y='{top+plot_h+18}' font-family='Arial' font-size='12' fill='#555'>{min_y:.4f}</text>",
        ]

        legend_x, legend_y = left + 10, 62
        for name, key, color in series:
            pts = []
            for i, row in enumerate(rows, start=1):
                v = row.get(key)
                if v is not None and v != "":
                    pts.append(f"{xp(i):.2f},{yp(float(v)):.2f}")
            if pts:
                parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{' '.join(pts)}'/>")
                parts.append(f"<rect x='{legend_x}' y='{legend_y-10}' width='12' height='12' fill='{color}'/>")
                parts.append(f"<text x='{legend_x+18}' y='{legend_y}' font-family='Arial' font-size='13' fill='#222'>{html.escape(name)}</text>")
                legend_y += 18

        parts.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

    # -------------------------------------------------------------------------
    # Step 4: draw graphs
    # -------------------------------------------------------------------------

    draw_svg(
        graphs_dir / "loss_curve.svg",
        "Reward Head — Training Loss",
        [("loss", "loss", "#7c3aed")],
    )

    draw_svg(
        graphs_dir / "predicted_vs_actual.svg",
        "Reward Head — Predicted vs Actual",
        [
            ("predicted", "predicted", "#2563eb"),
            ("actual",    "actual",    "#16a34a"),
        ],
    )

    # -------------------------------------------------------------------------
    # Step 5: write summary
    # -------------------------------------------------------------------------

    losses     = get_values("loss")
    predicted  = get_values("predicted")
    actual     = get_values("actual")

    # find latest checkpoint
    ckpt_dir  = Path(__file__).resolve().parent / "checkpoints"
    ckpts     = sorted(ckpt_dir.glob("reward_head_*.pt")) if ckpt_dir.exists() else []
    ckpt_name = ckpts[-1].name if ckpts else "none"

    summary_path = stats_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Reward Head Stats\n")
        f.write("=================\n\n")
        f.write(f"jsonl files: {len(list(data_dir.glob('*.jsonl')))}\n")
        f.write(f"rows:        {len(rows)}\n")
        f.write(f"checkpoint:  {ckpt_name}\n\n")
        if losses:
            f.write(f"loss first:  {losses[0]:.6f}\n")
            f.write(f"loss last:   {losses[-1]:.6f}\n")
            f.write(f"loss min:    {min(losses):.6f}\n")
            f.write(f"loss avg:    {sum(losses)/len(losses):.6f}\n\n")
        else:
            f.write("loss: no rows\n\n")
        if predicted and actual:
            errors = [abs(p - a) for p, a in zip(predicted, actual)]
            f.write(f"mean abs error: {sum(errors)/len(errors):.6f}\n")
            f.write(f"max abs error:  {max(errors):.6f}\n")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {graphs_dir / 'loss_curve.svg'}")
    print(f"wrote {graphs_dir / 'predicted_vs_actual.svg'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reward head stats")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    reward_head_stats(args.data_dir)
