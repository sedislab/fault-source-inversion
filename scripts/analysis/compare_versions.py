"""Side-by-side of two result versions on the metrics that survived the 2026-07-30 audit.

Reports margin-over-floor, not raw score. The floor is `zz_constant`, a scorer with no information about the
source; the previous headline metric (CvV) was retired because a degenerate scorer beat every real method on it,
so any table that does not show its own floor cannot be read."""
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True, help="e.g. results/franka_v4")
ap.add_argument("--new", required=True, help="e.g. results/franka_v6")
ap.add_argument("--split", default="in_distribution")
ap.add_argument("--out", default=None)
args = ap.parse_args()

def load(d, split):
    d = Path(d)
    hits = sorted(d.glob(f"*_{split}.csv"))
    if not hits:
        raise SystemExit(f"no *_{split}.csv under {d}")
    df = pd.read_csv(hits[0]).set_index("localizer")
    return df, hits[0].name

old, oname = load(args.old, args.split)
new, nname = load(args.new, args.split)
print(f"OLD {oname}\nNEW {nname}\n")

floor_o = old["hard_top1"].get("zz_constant", float("nan")) if "hard_top1" in old else float("nan")
floor_n = new["hard_top1"].get("zz_constant", float("nan"))
chance = new["chance"].dropna().iloc[0] if "chance" in new else float("nan")

rows = []
for k in sorted(set(old.index) | set(new.index)):
    o, n = old.loc[k] if k in old.index else None, new.loc[k] if k in new.index else None
    g = lambda s, c: (float(s[c]) if s is not None and c in s and pd.notna(s[c]) else float("nan"))
    rows.append({
        "localizer": k,
        "top1_old": g(o, "top1"), "top1_new": g(n, "top1"),
        "hard_old": g(o, "hard_top1"), "hard_new": g(n, "hard_top1"),
        "hard_sd_new": g(n, "hard_top1_sd"),
        "over_floor_new": g(n, "hard_top1") - floor_n,
    })
df = pd.DataFrame(rows).sort_values("hard_new", ascending=False)
txt = [f"# {args.split}: {args.old} -> {args.new}", "",
       f"chance = {chance:.3f} | zz_constant floor (new) = {floor_n:.3f}", "",
       df.round(3).to_markdown(index=False), "",
       "`over_floor_new` is the only number that supports a claim of superiority: it is hard-episode accuracy "
       "minus what a scorer with no information about the source achieves on the same episodes. A margin smaller "
       "than `hard_sd_new` is not a result.",
       "", "CvV is absent by design -- it was retired on 2026-07-30 after a scorer ranking nodes by NEGATIVE "
       "residual energy achieved CvV 1.000 while localizing almost nothing (top-1 0.065)."]
out = "\n".join(txt)
print(out)
if args.out:
    Path(args.out).write_text(out)
    print(f"\nsaved to {args.out}")
