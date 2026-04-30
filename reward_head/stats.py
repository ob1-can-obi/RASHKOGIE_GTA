"""
Reward head stats.

Reads reward_head/training_data/*.jsonl and writes:
- training_data/stats/reward_head_stats.csv
- training_data/stats/summary.txt
- training_data/graphs/reward_curve.svg
- training_data/graphs/reward_components.svg
"""

from pathlib import Path
import argparse
import csv
import html
import json


def reward_head_stats(data_dir=None):
    """
    Build CSV and SVG graphs for reward logs.
    """

    # -------------------------------------------------------------------------
    # Step 1: read reward JSONL rows
    # -------------------------------------------------------------------------

    data_dir = Path(data_dir or Path(__file__).resolve().parent / "training_data")
    stats_dir = data_dir / "stats"
    graphs_dir = data_dir / "graphs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_file"] = path.name
                row["_line"] = line_number
                rows.append(row)

    # -------------------------------------------------------------------------
    # Step 2: write flat CSV with reward components
    # -------------------------------------------------------------------------

    columns = [
        "source_file",
        "line",
        "global_index",
        "step",
        "ts",
        "next_ts",
        "reward",
        "distance",
        "previous_distance",
        "progress",
        "distance_term",
        "progress_term",
        "step_term",
        "collision",
        "collision_term",
        "offroad",
        "road_dist",
        "offroad_term",
        "goal_reached",
        "goal_term",
        "time_remaining",
        "time_term",
    ]

    csv_path = stats_dir / "reward_head_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            components = row.get("components", {})
            writer.writerow(
                {
                    "source_file": row.get("_file", ""),
                    "line": row.get("_line", ""),
                    "global_index": index,
                    "step": row.get("step", ""),
                    "ts": row.get("ts", ""),
                    "next_ts": row.get("next_ts", ""),
                    "reward": row.get("reward", ""),
                    "distance": components.get("distance", ""),
                    "previous_distance": components.get("previous_distance", ""),
                    "progress": components.get("progress", ""),
                    "distance_term": components.get("distance_term", ""),
                    "progress_term": components.get("progress_term", ""),
                    "step_term": components.get("step_term", ""),
                    "collision": components.get("collision", ""),
                    "collision_term": components.get("collision_term", ""),
                    "offroad": components.get("offroad", ""),
                    "road_dist": components.get("road_dist", ""),
                    "offroad_term": components.get("offroad_term", ""),
                    "goal_reached": components.get("goal_reached", ""),
                    "goal_term": components.get("goal_term", ""),
                    "time_remaining": components.get("time_remaining", ""),
                    "time_term": components.get("time_term", ""),
                }
            )

    # -------------------------------------------------------------------------
    # Step 3: draw simple SVG graphs
    # -------------------------------------------------------------------------

    def number(row, key):
        if key == "reward":
            value = row.get("reward", None)
        else:
            value = row.get("components", {}).get(key, None)
        if value is None or value == "":
            return None
        return float(value)

    def metric(key):
        values = []
        for row in rows:
            value = number(row, key)
            if value is not None:
                values.append(value)
        return values

    def draw_svg(path, title, series):
        width = 960
        height = 520
        left = 70
        right = 30
        top = 45
        bottom = 70
        plot_w = width - left - right
        plot_h = height - top - bottom

        all_points = []
        for name, key, color in series:
            for index, row in enumerate(rows, start=1):
                value = number(row, key)
                if value is not None:
                    all_points.append((index, value))

        if not all_points:
            with open(path, "w", encoding="utf-8") as file:
                file.write(
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
                    f"<text x='24' y='32' font-family='Arial' font-size='22'>{html.escape(title)}</text>"
                    f"<text x='24' y='72' font-family='Arial' font-size='16'>No rows yet</text>"
                    "</svg>"
                )
            return

        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        if max_x == min_x:
            max_x = min_x + 1
        if max_y == min_y:
            max_y = min_y + 1.0

        def x_pos(x_value):
            return left + ((x_value - min_x) / (max_x - min_x)) * plot_w

        def y_pos(y_value):
            return top + (1.0 - ((y_value - min_y) / (max_y - min_y))) * plot_h

        parts = []
        parts.append(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        )
        parts.append("<rect width='100%' height='100%' fill='white'/>")
        parts.append(
            f"<text x='24' y='32' font-family='Arial' font-size='22' fill='#111'>{html.escape(title)}</text>"
        )
        parts.append(
            f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#888'/>"
        )
        parts.append(
            f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#888'/>"
        )
        parts.append(
            f"<text x='{left}' y='{height - 25}' font-family='Arial' font-size='13' fill='#555'>samples: {len(rows)}</text>"
        )
        parts.append(
            f"<text x='{left}' y='{top - 10}' font-family='Arial' font-size='12' fill='#555'>{max_y:.4f}</text>"
        )
        parts.append(
            f"<text x='{left}' y='{top + plot_h + 18}' font-family='Arial' font-size='12' fill='#555'>{min_y:.4f}</text>"
        )

        legend_x = left + 10
        legend_y = 62
        for name, key, color in series:
            points = []
            for index, row in enumerate(rows, start=1):
                value = number(row, key)
                if value is not None:
                    points.append(f"{x_pos(index):.2f},{y_pos(value):.2f}")
            if points:
                parts.append(
                    f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{' '.join(points)}'/>"
                )
                parts.append(
                    f"<rect x='{legend_x}' y='{legend_y - 10}' width='12' height='12' fill='{color}'/>"
                )
                parts.append(
                    f"<text x='{legend_x + 18}' y='{legend_y}' font-family='Arial' font-size='13' fill='#222'>{html.escape(name)}</text>"
                )
                legend_y += 18

        parts.append("</svg>")
        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(parts))

    draw_svg(
        graphs_dir / "reward_curve.svg",
        "Reward Curve",
        [("reward", "reward", "#7c3aed")],
    )

    draw_svg(
        graphs_dir / "reward_components.svg",
        "Reward Components",
        [
            ("distance term", "distance_term", "#dc2626"),
            ("progress term", "progress_term", "#16a34a"),
            ("step term", "step_term", "#6b7280"),
            ("collision term", "collision_term", "#111827"),
            ("offroad term", "offroad_term", "#ea580c"),
            ("goal term", "goal_term", "#2563eb"),
            ("time term", "time_term", "#0891b2"),
        ],
    )

    # -------------------------------------------------------------------------
    # Step 4: write summary
    # -------------------------------------------------------------------------

    rewards = metric("reward")
    collision_rows = 0
    offroad_rows = 0
    goal_rows = 0
    for row in rows:
        if float(row.get("components", {}).get("collision", 0.0)) > 0.0:
            collision_rows += 1
        if float(row.get("components", {}).get("offroad", 0.0)) > 0.0:
            offroad_rows += 1
        if float(row.get("components", {}).get("goal_reached", 0.0)) > 0.0:
            goal_rows += 1

    summary_path = stats_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Reward Head Stats\n")
        file.write("=================\n\n")
        file.write(f"jsonl files: {len(list(data_dir.glob('*.jsonl')))}\n")
        file.write(f"rows: {len(rows)}\n")
        file.write("checkpoint: none, formula-based\n\n")
        if rewards:
            file.write(f"reward first: {rewards[0]:.6f}\n")
            file.write(f"reward last:  {rewards[-1]:.6f}\n")
            file.write(f"reward change:{rewards[-1] - rewards[0]:.6f}\n")
            file.write(f"reward avg:   {sum(rewards) / len(rewards):.6f}\n")
            file.write(f"reward min:   {min(rewards):.6f}\n")
            file.write(f"reward max:   {max(rewards):.6f}\n\n")
        else:
            file.write("reward: no rows\n\n")
        file.write(f"collision rows: {collision_rows}\n")
        file.write(f"offroad rows:   {offroad_rows}\n")
        file.write(f"goal rows:      {goal_rows}\n")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {graphs_dir / 'reward_curve.svg'}")
    print(f"wrote {graphs_dir / 'reward_components.svg'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reward head stats")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    reward_head_stats(args.data_dir)

