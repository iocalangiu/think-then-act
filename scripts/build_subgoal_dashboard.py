"""
build_subgoal_dashboard.py

Local (no Modal) — stitches artifacts/subgoal_dashboard_template.html (the
dashboard's HTML/CSS/JS, data-free) together with a freshly aggregated
subgoal-distance-grid JSON (see aggregate_subgoal_distance_grid.py) into one
self-contained HTML file ready to hand to the Artifact tool for publishing.

Full pipeline to refresh the live dashboard after retraining a subgoal:
    modal run scripts/wandb_subgoal_distance_grid.py
    python3 scripts/aggregate_subgoal_distance_grid.py
    python3 scripts/build_subgoal_dashboard.py
    # then: Artifact(file_path="artifacts/subgoal_dashboard.html",
    #                 url="<the existing dashboard's URL — see project memory
    #                      subgoal_distance_dashboard.md>")

Run with:
    python3 scripts/build_subgoal_dashboard.py
    python3 scripts/build_subgoal_dashboard.py --template artifacts/subgoal_dashboard_template.html \
        --data artifacts/subgoal_distance_grid_agg.json --out artifacts/subgoal_dashboard.html
"""

import argparse
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="artifacts/subgoal_dashboard_template.html")
    parser.add_argument("--data", default="artifacts/subgoal_distance_grid_agg.json")
    parser.add_argument("--out", default="artifacts/subgoal_dashboard.html")
    args = parser.parse_args()

    template = open(args.template).read()
    data = open(args.data).read().strip()

    placeholder = "const DATA = /*__DATA__*/;"
    if placeholder not in template:
        raise ValueError(
            f"Template {args.template!r} doesn't contain the expected "
            f"{placeholder!r} placeholder — was it edited directly with data "
            f"embedded instead of through this template? Re-derive the "
            f"template from the template's own DATA line pattern before "
            f"editing further."
        )

    html = template.replace(placeholder, f"const DATA = {data};")

    with open(args.out, "w") as f:
        f.write(html)

    print(f"Built {args.out} from {args.template} + {args.data} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
