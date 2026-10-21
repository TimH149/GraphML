"""
WHAT IT DOES:
----------------------
Tunes CorGCN's k_num hyperparameter on a single split of a dataset. (Other hyperparameters stay fixed)
Best k_num is selected based on validation micro-AUC only, ignoring test numbers.

HOW TO RUN:
----------------------
(assuming data folder is in the same parent directory, one level down from the root.)
For e.g. DBLP run using 
python tune_k.py --data_name mf_dblp --mf_root ../data/ 

For e.g. Synthetic025, run using
python tune_k.py --data_name mf_hyper --mf_root ../data/ --extra --homo_level homo025

"""

import argparse
import os
import re
import subprocess
import sys

LINE = re.compile(
    r"Eval macro AP: ([\d.eE+-]+), eval micro AP: ([\d.eE+-]+), "
    r"eval macro AUC: ([\d.eE+-]+), eval micro AUC: ([\d.eE+-]+)"
)


def parse_line(line):
    m = LINE.search(line)
    if not m:
        return None
    macro_ap, micro_ap, macro_auc, micro_auc = map(float, m.groups())
    return dict(macro_ap=macro_ap, micro_ap=micro_ap, macro_auc=macro_auc, micro_auc=micro_auc)


def run_one(cmd, log_base):
    """Run one training process, stream progress, return (returncode, best_val, epochs_run, finished_training)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    best, n, finished = None, 0, False
    with open(log_base + ".err.log", "w", encoding="utf-8") as ef, \
         open(log_base + ".out.log", "w", encoding="utf-8") as of:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=ef, text=True,
                             encoding="utf-8", errors="replace", env=env, bufsize=1)
        for line in p.stdout:
            of.write(line)
            of.flush()
            if "Train done!" in line:
                finished = True
            r = parse_line(line)
            if r:
                n += 1
                if best is None or r["micro_auc"] > best["micro_auc"]:   # first max, like EarlyStopper
                    best = dict(r, epoch=n - 1)
                if n % 5 == 0:
                    print(f"   epoch {n:4d}: val micro-AUC {r['micro_auc']:.4f} (best {best['micro_auc']:.4f})", flush=True)
        p.wait()
    return p.returncode, best, n, finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="run_CorGCN.py")
    ap.add_argument("--data_name", default="mf_dblp")
    ap.add_argument("--mf_root", required=True)
    ap.add_argument("--split", default="split_0.pt")
    ap.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--lr", default="0.001")
    ap.add_argument("--bs", default="1024")
    ap.add_argument("--early_epochs", default="30", help="short patience, only for ranking k")
    ap.add_argument("--num_epochs", default="50", help="hard cap per run, only for ranking k")
    ap.add_argument("--gpu_id", default="0")
    ap.add_argument("--select", choices=["micro_auc", "macro_ap"], default="micro_auc")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="anything else to forward, e.g. --extra --cluster_num 10")
    a = ap.parse_args()

    os.makedirs("tune_logs", exist_ok=True)
    results = {}   # k -> (status, best, epochs_run)
    for k in a.k_values:
        param_path = f"best_param_tune/k{k}"
        os.makedirs(param_path, exist_ok=True)   # separate checkpoints per k
        cmd = [sys.executable, "-u", a.script, "--model_name", f"corgcn_k{k}", "--data_name", a.data_name,
               "--mf_root", a.mf_root, "--split_names", a.split, "--use_gpu", "--gpu_id", a.gpu_id,
               "--lr", a.lr, "--bs", a.bs, "--k_num", str(k), "--early_epochs", a.early_epochs,
               "--num_epochs", a.num_epochs, "--param_path", param_path] + a.extra
        print(f"\n=== k_num={k} ===", flush=True)
        rc, best, n, finished = run_one(cmd, f"tune_logs/{a.data_name}_k{k}")

        if best is None:
            print(f"no validation lines parsed (exit {rc}); see tune_logs/{a.data_name}_k{k}.*.log")
            results[k] = ("no data", None, n)
            continue
        if rc == 0:
            status = "ok"
        elif finished or n >= int(a.num_epochs):
            status = "crashed after training (val usable)"
        else:
            status = f"crashed during training at epoch {n} (partial)"
        results[k] = (status, best, n)
        print(f"-> {status}: val micro-AUC={best['micro_auc']:.4f}  val macro-AP={best['macro_ap']:.4f} "
              f"(best epoch {best['epoch']}, {n} epochs run, exit {rc})", flush=True)

    print("\n k_num | val micro-AUC | val macro-AP | best ep | epochs | status")
    for k, (status, b, n) in sorted(results.items()):
        if b:
            print(f" {k:5d} | {b['micro_auc']:13.4f} | {b['macro_ap']:12.4f} | {b['epoch']:7d} | {n:6d} | {status}")
        else:
            print(f" {k:5d} | {'-':>13} | {'-':>12} | {'-':>7} | {n:6d} | {status}")

    usable = {k: b for k, (s, b, n) in results.items() if b and (s == "ok" or s.startswith("crashed after"))}
    if not usable:
        print("\nNo usable run -> fix the crash first (see tune_logs/*.out.log and *.err.log).")
        return
    best_k = max(usable, key=lambda k: usable[k][a.select])
    print(f"\nBest k_num by validation {a.select}: {best_k}")


if __name__ == "__main__":
    main()
