"""Shared GitHub Actions interval reader and timing reports (called by shell tools)."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import re
from statistics import median
import sys


def interval(start, end):
    """Missing, invalid or reversed endpoints are unavailable; equal ones are 0s."""
    if not start or not end:
        return None
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        d = (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds()
    except (TypeError, ValueError):
        return None
    return d if d >= 0 else None


def duration(item):
    return interval(item.get("started_at"), item.get("completed_at"))


def human(s):
    m, s = divmod(int(round(s)), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def label_for(job, total):
    if total is not None:
        return human(total)
    status = job.get("status")
    if status and status != "completed":
        return "(" + str(status) + ")"
    return "(no time)"


def times(jobs, threshold):
    for job in jobs:
        total = duration(job)
        label = label_for(job, total)
        conclusion = job.get("conclusion")
        tail = "" if conclusion in (None, "success") else "  [" + conclusion + "]"
        print(f'{label:>8}  {job["name"]}{tail}')
        for step in job.get("steps") or []:
            d = duration(step)
            if d is not None and d > threshold:
                print(f'{human(d):>12}    {step["name"]}')


def stats(values, scale=60):
    if not values:
        return "0 unavailable"
    return f"{len(values)} {min(values)/scale:.1f} {median(values)/scale:.1f} {max(values)/scale:.1f}"


def durations(jobs, header, job_re, step_re):
    groups = defaultdict(list)
    excluded = Counter()
    for job in jobs:
        if job_re and not job_re.search(job["name"]):
            continue
        groups[(job["name"], job.get("conclusion") or "null")].append(job)
    if not groups:
        raise ValueError("no jobs matched the job selector")
    if not step_re:
        rows = []
        for (name, conclusion), group in sorted(groups.items()):
            values = [d for job in group if (d := duration(job)) is not None]
            excluded["job(s) with no usable start/finish time"] += len(group) - len(values)
            if values:
                rows.append((f"{name} [{conclusion}]", values))
        if not rows:
            raise ValueError("no job had a usable start/finish time")
        print(header + "\n")
        print(f'{"job [conclusion]":52} {"n":>5} {"min":>8} {"median":>8} {"max":>8}')
        for key, values in rows:
            print(f"{key:52} {len(values):5} {min(values)/60:8.1f} {median(values)/60:8.1f} {max(values)/60:8.1f}")
    else:
        print(header + "\n")
        print(f"job selector: {job_re.pattern if job_re else '(all)'}; step selector: {step_re.pattern}")
        print("distributions: n min median max (minutes; share in percent)")
        for (name, conclusion), group in sorted(groups.items()):
            samples = defaultdict(list)
            counts = Counter()
            names = set()
            for job in group:
                selected = [s for s in job.get("steps") or [] if step_re.search(s["name"])]
                names.update(s["name"] for s in selected)
                if not selected:
                    counts["missing"] += 1
                    continue
                selected = [s for s in selected if s.get("conclusion") != "skipped"]
                if not selected:
                    counts["unusable"] += 1
                    continue
                values = [duration(s) for s in selected]
                if any(d is None for d in values):
                    counts["unusable"] += 1
                    continue
                step = sum(values)
                samples["step"].append(step)
                job_time = duration(job)
                if job_time is None or job_time <= 0 or step > job_time:
                    counts["unpaired"] += 1
                    continue
                samples["job"].append(job_time)
                samples["rest"].append(job_time - step)
                samples["share"].append(100 * step / job_time)
            print(f"\n{name} [{conclusion}] jobs={len(group)} missing={counts['missing']} unusable={counts['unusable']} unpaired={counts['unpaired']}")
            print("  matched steps: " + (", ".join(sorted(names)) or "(none)"))
            for metric in ("step", "job", "rest", "share"):
                print(f"  {metric}: {stats(samples[metric], 1 if metric == 'share' else 60)}")
    for label, count in excluded.items():
        if count:
            print(f"ci-durations.sh: skipped {count} {label}", file=sys.stderr)


def regex(value):
    try:
        return re.compile(value)
    except re.error as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("times").add_argument("threshold", type=int)
    for mode in ("durations", "validate"):
        command = sub.add_parser(mode)
        if mode == "durations":
            command.add_argument("header")
        command.add_argument("--job", type=regex)
        command.add_argument("--step", type=regex)
    args = parser.parse_args()
    if args.mode == "validate":
        return
    jobs = [json.loads(line) for line in sys.stdin if line.strip()]
    if args.mode == "times":
        times(jobs, args.threshold)
    else:
        durations(jobs, args.header, args.job, args.step)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, KeyError) as error:
        print(f"ci-timing: {error}", file=sys.stderr)
        sys.exit(1)
