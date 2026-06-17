#!/usr/bin/env python3
"""Summarize judge results into tables with confidence intervals.

CI method (default): bootstrap over per-sample means.
  1. For each benchmark sample (one interruption point), average the metric
     across the E epochs the model was run for. This is the per-sample mean.
  2. Bootstrap-resample the N per-sample means with replacement (1000 iters).
  3. Report 2.5/97.5 percentiles of the bootstrap distribution.

This is the textbook recipe for benchmark CIs with repeated measurements
(see e.g. Lewis 1980; common practice in modern ML benchmark papers). It
isolates the question we actually care about — "would this rank hold on a
different set of N interruption points drawn from the same population?" —
from decoding-noise variance, which is folded into each per-sample mean.

Other options:
  --ci epoch_t : legacy across-epoch t-distribution CI (df = n_epochs - 1),
                 kept only for back-compat / sanity checks.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict


N_BOOT = 1000
SEED = 0


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# CI helpers
# ---------------------------------------------------------------------------

def _epoch_t_ci(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(variance / n)
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    margin = t_crit * stderr
    return mean, mean - margin, mean + margin


def _sample_key(r):
    """The benchmark sample identity (unique interruption point), independent of epoch."""
    return (r.get("conversation_id"), r.get("sample_id") or r.get("turn_id"))


def _bootstrap_ci(rows, value_fn, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap CI over per-sample means.

    Standard benchmark-eval recipe:
      1. Group rows by benchmark sample (interruption point).
      2. Within each sample, average across epochs.
      3. Bootstrap-resample the N per-sample means with replacement (N times).
      4. Report mean and 2.5-97.5 percentile of the resample distribution.

    This isolates sample-to-sample variability (the generalization question)
    from decoding noise (the within-sample epoch variance, which is absorbed
    into the per-sample mean before resampling).

    rows: list of judgment dicts.
    value_fn: dict -> float | None. None entries are dropped.
    """
    by_sample = defaultdict(list)
    for r in rows:
        v = value_fn(r)
        if v is not None:
            by_sample[_sample_key(r)].append(v)
    sample_means = [sum(vs) / len(vs) for vs in by_sample.values()]
    n = len(sample_means)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = sum(sample_means) / n
    if n == 1:
        return point, point, point
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += sample_means[rng.randrange(n)]
        samples.append(s / n)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot) - 1]
    return point, lo, hi


# ---------------------------------------------------------------------------
# Per-model metrics
# ---------------------------------------------------------------------------

def compute_metrics_with_ci(results, ci_method="bootstrap"):
    n_total = len(results)
    n_epochs = len({r.get("epoch", 0) for r in results})

    if ci_method == "bootstrap":
        tf = _bootstrap_ci(results, lambda r: r.get("tf_win_score"))
        rq = _bootstrap_ci(results, lambda r: int(r["rq_pass"]) if r.get("rq_pass") is not None else None)
        avg = _bootstrap_ci(results, lambda r: r.get("rq_score"))
    else:  # epoch_t
        by_epoch = defaultdict(list)
        for r in results:
            by_epoch[r.get("epoch", 0)].append(r)
        tf_pe, rq_pe, avg_pe = [], [], []
        for ep_rows in by_epoch.values():
            tf_vals = [r["tf_win_score"] for r in ep_rows if r.get("tf_win_score") is not None]
            rq_vals = [int(r["rq_pass"]) for r in ep_rows if r.get("rq_pass") is not None]
            avg_vals = [r["rq_score"] for r in ep_rows if r.get("rq_score") is not None]
            if tf_vals:
                tf_pe.append(sum(tf_vals) / len(tf_vals))
            if rq_vals:
                rq_pe.append(sum(rq_vals) / len(rq_vals))
            if avg_vals:
                avg_pe.append(sum(avg_vals) / len(avg_vals))
        tf = _epoch_t_ci(tf_pe)
        rq = _epoch_t_ci(rq_pe)
        avg = _epoch_t_ci(avg_pe)

    return {
        "n": n_total,
        "epochs": n_epochs,
        "tf": tf,
        "rq": rq,
        "avg": avg,
    }


def compute_per_type_with_ci(results, ci_method="bootstrap"):
    by_type = defaultdict(list)
    for r in results:
        if r.get("rq_pass") is not None:
            by_type[r["interruption_type"]].append(r)

    out = {}
    if ci_method == "bootstrap":
        for itype, rows in sorted(by_type.items()):
            out[itype] = _bootstrap_ci(rows, lambda r: int(r["rq_pass"]))
    else:
        for itype, rows in sorted(by_type.items()):
            by_ep = defaultdict(list)
            for r in rows:
                by_ep[r.get("epoch", 0)].append(int(r["rq_pass"]))
            per_ep = [sum(v) / len(v) for v in by_ep.values() if v]
            out[itype] = _epoch_t_ci(per_ep)
    return out


# ---------------------------------------------------------------------------
# Formatting / printing
# ---------------------------------------------------------------------------

def fmt_ci(vals):
    mean, lo, hi = vals
    if lo == hi:
        return "%.3f" % mean
    return "%.3f [%.3f, %.3f]" % (mean, lo, hi)


def fmt_ci_short(vals):
    mean, lo, hi = vals
    if lo == hi:
        return "%.3f" % mean
    halfwidth = (hi - lo) / 2
    return "%.3f +/- %.3f" % (mean, halfwidth)


TYPES = ["normal", "impatient", "correction", "topic_switch", "filler", "pushback"]


def print_table(label, directory, ci_method, min_results=50):
    print("=== %s ===" % label)
    print("%-35s %5s %3s  %-21s %-21s %-21s" % ("Model", "N", "Ep", "TF Win Rate", "RQ Pass Rate", "RQ Avg"))
    print("-" * 115)
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".jsonl"):
            continue
        path = os.path.join(directory, f)
        results = load_results(path)
        if len(results) < min_results:
            print("%-35s %5d  -- (too few)" % (f.replace(".jsonl", ""), len(results)))
            continue
        name = f.replace(".jsonl", "")
        m = compute_metrics_with_ci(results, ci_method)
        print("%-35s %5d  %d  %-21s %-21s %-21s" % (
            name, m["n"], m["epochs"],
            fmt_ci(m["tf"]), fmt_ci(m["rq"]), fmt_ci(m["avg"]),
        ))
    print()


def print_per_type(label, directory, ci_method, min_results=50):
    print("=== %s (Per-Type RQ Pass Rate) ===" % label)
    header = "%-35s" % "Model" + "".join("%-24s" % t for t in TYPES)
    print(header)
    print("-" * (35 + 24 * len(TYPES)))
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".jsonl"):
            continue
        path = os.path.join(directory, f)
        results = load_results(path)
        if len(results) < min_results:
            continue
        name = f.replace(".jsonl", "")
        pt = compute_per_type_with_ci(results, ci_method)
        row = "%-35s" % name
        for t in TYPES:
            if t in pt:
                row += "%-24s" % fmt_ci_short(pt[t])
            else:
                row += "%-24s" % "--"
        print(row)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--ci", choices=["bootstrap", "epoch_t"], default="bootstrap",
                        help="CI method: bootstrap over per-sample means (default) or across-epoch t-CI (legacy).")
    args = parser.parse_args()

    print_table("Audio + Tags (Main)", args.results_dir, args.ci)
    print_per_type("Audio + Tags (Main)", args.results_dir, args.ci)

    notag_dir = os.path.join(args.results_dir, "notag")
    if os.path.isdir(notag_dir):
        print_table("Audio + No-Tag (Ablation)", notag_dir, args.ci)

    textonly_dir = os.path.join(args.results_dir, "textonly")
    if os.path.isdir(textonly_dir):
        print_table("Text-Only + Tags", textonly_dir, args.ci)
