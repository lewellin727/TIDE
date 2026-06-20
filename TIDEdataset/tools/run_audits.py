
import os
import re
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(tool, lake, suffix):
    cmd = [sys.executable, os.path.join(ROOT, "tools", f"{tool}.py"), "--suffix", suffix, "--lake", lake]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout, r.stderr


def _pq_counts(out):
    m = re.search(r"fully-pass=(\d+) \| with-issues=(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _struct_metric_lines(out):
    return [ln.rstrip() for ln in out.splitlines()
            if ln.strip().startswith(("(", "depth", "-- "))]


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else cfg("output_suffix", "")
    split = f"{cfg('dataset')}{suffix}"
    ts = time.strftime("%Y%m%d_%H%M%S")

    runs = []   # (tool, lake, stdout, stderr)
    for lake in ("lakes", "aug_lakes"):
        for tool in ("audit_per_query", "audit_reachability"):
            out, err = _run(tool, lake, suffix)
            runs.append((tool, lake, out, err))

    # print the full audit detail to stdout (redirect the command to capture it)
    print(f"AUDIT  split={split}  time={ts}  (per-query authoritative + structural, lakes & aug_lakes)")
    for tool, lake, out, err in runs:
        print(f"\n{'#' * 90}\n# {tool}   --lake {lake}\n{'#' * 90}\n{out}")
        if err.strip():
            print(f"[stderr]\n{err}")

    # ---- printed verdict ----
    issues_total = 0
    for tool, lake, out, _ in runs:
        if tool == "audit_per_query":
            p, i = _pq_counts(out)
            issues_total += (i or 0)
            print(f"  per-query   @ {lake:9}: fully-pass={p}  with-issues={i}")
    struct = {lake: out for tool, lake, out, _ in runs if tool == "audit_reachability"}
    identical = _struct_metric_lines(struct.get("lakes", "")) == _struct_metric_lines(struct.get("aug_lakes", "x"))
    print(f"  structural  lakes == aug_lakes : {'YES (augmentation preserves all properties)' if identical else 'NO  <-- INVESTIGATE'}")
    ok = (issues_total == 0) and identical
    print(f"  VERDICT: {'ALL PASS ✓' if ok else 'ISSUES — inspect the log'}"
          + ("" if ok else f"  (per-query issues={issues_total}, struct_identical={identical})"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
