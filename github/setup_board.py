#!/usr/bin/env python3
"""Create the GitHub repo, labels, milestones, issues and Project board for the capstone (P1 / T01).

Requirements: GitHub CLI (`gh`) logged in with the `project` scope:
    gh auth login
    gh auth refresh -s project

Run from the repository root:
    python github/setup_board.py --dry-run          # print every command, change nothing
    python github/setup_board.py                    # do it

Configuration (environment variables, all optional):
    REPO_NAME    default weather-food-orders
    START_DATE   Monday of week 1, default 2026-10-05. Due date of a task in week N = Friday of week N.
                 Replace with your real schedule (Review 1 = end of W4, Review 2 = end of W8).
    P1_USER ... P5_USER   GitHub usernames of the five members. Issues are assigned to them;
                 if a variable is missing the issue is created unassigned and a warning is printed
                 (the rubric requires every task to have an owner, so fill these in).

What it cannot do from the CLI: rename the built-in Status options to the five rubric columns if the
GraphQL call below is refused. The script prints the manual steps in that case (about 30 seconds in the UI).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COLUMNS = ["Backlog", "To Do", "In Progress", "Review/Testing", "Completed"]

ISSUE_TEMPLATE = """---
name: Task
about: One planned task from the project plan
labels: ''
---

**Task ID:** T__
**Owner:**
**Due:**
**Rubric item:**

## What
<!-- one or two sentences -->

## Evidence to attach (required before moving to Completed)
<!-- link to the PR, notebook cell, chart, dataset or slide -->

## Definition of done
- [ ] Branch named `tNN-short-name`, never committed to `main`
- [ ] PR opened with `Closes #<issue>`
- [ ] A different teammate approved the PR
- [ ] Evidence field on the project card filled in
"""

PR_TEMPLATE = """Closes #

## What changed

## How to check it
<!-- command to run, notebook cell, or file to open -->

## Evidence
<!-- chart, table or output that shows it works -->
"""


class Runner:
    def __init__(self, dry: bool):
        self.dry = dry

    def run(self, cmd: list[str], check: bool = True, capture: bool = False, input_: str | None = None) -> str:
        print("$", " ".join(shlex.quote(c) for c in cmd))
        if self.dry:
            return ""
        p = subprocess.run(cmd, text=True, capture_output=capture, input=input_)
        if check and p.returncode != 0:
            sys.exit(f"command failed ({p.returncode}): {p.stderr if capture else ''}")
        return p.stdout if capture else ""

    def gh_json(self, cmd: list[str]) -> dict | list:
        out = self.run(cmd, capture=True)
        return json.loads(out) if out.strip() else {}


def due_date(start: date, week: int) -> str:
    return (start + timedelta(days=7 * (week - 1) + 4)).isoformat()   # Friday of that week


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-repo", action="store_true", help="repo already exists and is pushed")
    args = ap.parse_args()
    r = Runner(args.dry_run)

    repo_name = os.environ.get("REPO_NAME", "weather-food-orders")
    start = date.fromisoformat(os.environ.get("START_DATE", "2026-10-05"))
    users = {f"P{i}": os.environ.get(f"P{i}_USER", "") for i in range(1, 6)}
    for k, v in users.items():
        if not v:
            print(f"WARNING: {k}_USER not set, {k}'s issues will be unassigned")

    rows = list(csv.DictReader(open(HERE / "issues.csv", encoding="utf-8")))
    assert len(rows) == 29, "expected 29 tasks"

    owner = "@me" if args.dry_run else r.run(["gh", "api", "user", "--jq", ".login"], capture=True).strip()
    full = f"{owner}/{repo_name}" if owner != "@me" else f"<you>/{repo_name}"

    # ---- 1. templates + repo ------------------------------------------------------------------
    (ROOT / ".github/ISSUE_TEMPLATE").mkdir(parents=True, exist_ok=True)
    (ROOT / ".github/ISSUE_TEMPLATE/task.md").write_text(ISSUE_TEMPLATE, encoding="utf-8")
    (ROOT / ".github/pull_request_template.md").write_text(PR_TEMPLATE, encoding="utf-8")
    gi = ROOT / ".gitignore"
    if not gi.exists():
        gi.write_text(".ipynb_checkpoints/\n__pycache__/\n*.pyc\n.venv/\n.DS_Store\n", encoding="utf-8")

    if not args.skip_repo:
        if not (ROOT / ".git").exists():
            r.run(["git", "-C", str(ROOT), "init", "-b", "main"])
        r.run(["git", "-C", str(ROOT), "add", "-A"])
        r.run(["git", "-C", str(ROOT), "commit", "-m", "Initial project structure, weather fetcher, dataset docs and forecasting notebook"], check=False)
        r.run(["gh", "repo", "create", repo_name, "--private", "--source", str(ROOT), "--push"], check=False)

    # ---- 2. collaborators, labels, milestones --------------------------------------------------
    for k, u in users.items():
        if u and u != owner:
            r.run(["gh", "api", "-X", "PUT", f"repos/{full}/collaborators/{u}", "-f", "permission=push"], check=False)

    labels = {"R1": "1d76db", "R2": "e99695", **{f"P{i}": c for i, c in zip(range(1, 6), ["5319e7", "0e8a16", "fbca04", "d93f0b", "006b75"])}}
    for name, color in labels.items():
        r.run(["gh", "label", "create", name, "--color", color, "--force", "-R", full])

    ms = {"R1": ("Review 1", due_date(start, 4)), "R2": ("Review 2", due_date(start, 8))}
    for _, (title, due) in ms.items():
        r.run(["gh", "api", f"repos/{full}/milestones", "-f", f"title={title}", "-f", f"due_on={due}T23:59:59Z"], check=False)

    # ---- 3. branch protection on main -----------------------------------------------------------
    protection = {
        "required_status_checks": None,
        "enforce_admins": False,
        "required_pull_request_reviews": {"required_approving_review_count": 1, "dismiss_stale_reviews": True},
        "restrictions": None,
    }
    r.run(["gh", "api", "-X", "PUT", f"repos/{full}/branches/main/protection", "--input", "-"],
          check=False, input_=json.dumps(protection))

    # ---- 4. project board -----------------------------------------------------------------------
    proj = r.gh_json(["gh", "project", "create", "--owner", "@me", "--title", "Weather-Driven Food Orders", "--format", "json"]) or {}
    pnum, pid = str(proj.get("number", "<N>")), proj.get("id", "<PROJECT_ID>")
    r.run(["gh", "project", "link", pnum, "--owner", "@me", "--repo", full], check=False)

    for fname, ftype, opts in [("Due date", "DATE", None), ("Review", "SINGLE_SELECT", "R1,R2"),
                               ("Points", "NUMBER", None), ("Rubric item", "TEXT", None), ("Evidence", "TEXT", None)]:
        cmd = ["gh", "project", "field-create", pnum, "--owner", "@me", "--name", fname, "--data-type", ftype]
        if opts:
            cmd += ["--single-select-options", opts]
        r.run(cmd, check=False)

    fields = {f["name"]: f for f in (r.gh_json(["gh", "project", "field-list", pnum, "--owner", "@me", "--format", "json"]) or {}).get("fields", [])}

    # Status options -> the five rubric columns (GraphQL; falls back to a manual instruction)
    status = fields.get("Status")
    if status:
        mutation = ("mutation($f:ID!,$opts:[ProjectV2SingleSelectFieldOptionInput!]!){updateProjectV2Field(input:"
                    "{fieldId:$f,singleSelectOptions:$opts}){projectV2Field{... on ProjectV2SingleSelectField{options{id name}}}}}")
        opts = [{"name": c, "color": "GRAY", "description": ""} for c in COLUMNS]
        payload = json.dumps({"query": mutation, "variables": {"f": status["id"], "opts": opts}})
        r.run(["gh", "api", "graphql", "--input", "-"], check=False, input_=payload)
        fields = {f["name"]: f for f in r.gh_json(["gh", "project", "field-list", pnum, "--owner", "@me", "--format", "json"]).get("fields", [])}
        got = [o["name"] for o in fields.get("Status", {}).get("options", [])]
        if got != COLUMNS and not args.dry_run:
            print("\nMANUAL STEP: open the project > Status field settings and rename the options to exactly:\n  "
                  + "  |  ".join(COLUMNS) + "\nThen re-run with --skip-repo (issues are only created once; see below).\n")

    def opt_id(field: str, name: str):
        for o in fields.get(field, {}).get("options", []):
            if o["name"] == name:
                return o["id"]
        return "<OPTION_ID>"

    fid = lambda n: fields.get(n, {}).get("id", f"<{n}_FIELD_ID>")

    # ---- 5. issues + cards ----------------------------------------------------------------------
    for t in rows:
        body = (f"**Task {t['id']}** · Owner {t['owner']} · {t['review']} · {t['points']} pts · due week {t['week']}\n\n"
                f"{t['description']}\n\n**Rubric item:** {t['rubric_item']}\n**Evidence to attach:** {t['evidence']}\n\n"
                "Definition of done: PR merged with `Closes #<this issue>`, approved by a different teammate, Evidence field filled in.")
        cmd = ["gh", "issue", "create", "-R", full, "--title", f"{t['id']} {t['title']}", "--body", body,
               "--label", f"{t['review']},{t['owner']}", "--milestone", ms[t["review"]][0]]
        if users[t["owner"]]:
            cmd += ["--assignee", users[t["owner"]]]
        url = r.run(cmd, capture=True).strip() or f"https://github.com/{full}/issues/<n>"
        item = r.gh_json(["gh", "project", "item-add", pnum, "--owner", "@me", "--url", url, "--format", "json"]) or {}
        iid = item.get("id", "<ITEM_ID>")
        base = ["gh", "project", "item-edit", "--id", iid, "--project-id", pid]
        r.run(base + ["--field-id", fid("Due date"), "--date", due_date(start, int(t["week"]))], check=False)
        r.run(base + ["--field-id", fid("Points"), "--number", t["points"]], check=False)
        r.run(base + ["--field-id", fid("Rubric item"), "--text", t["rubric_item"]], check=False)
        r.run(base + ["--field-id", fid("Review"), "--single-select-option-id", opt_id("Review", t["review"])], check=False)
        r.run(base + ["--field-id", fid("Status"), "--single-select-option-id", opt_id("Status", "Backlog")], check=False)

    print("\nDone. Next: enable the built-in project workflows (Item added -> Backlog, PR merged -> Completed, "
          "Item closed -> Completed) under Project > Workflows, and add a Board view grouped by Status.")


if __name__ == "__main__":
    main()
