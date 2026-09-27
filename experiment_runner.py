"""
Experiment Runner — Orchestrator
Loops through all models and methods from config.
Each combination runs as a completely separate subprocess —
when it exits ALL memory is returned to the OS, guaranteed
clean slate for every method.
After all runs complete, generates the cross-method comparison chart.
"""

import csv
import subprocess
import sys
import time
import yaml
from pathlib import Path


def drop_caches():
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        print("[MEMORY] Caches dropped.")
    except PermissionError:
        print("[MEMORY] Could not drop caches — run as sudo.")


def print_memory():
    try:
        import subprocess as sp
        result = sp.run(["free", "-h"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"[MEMORY] Could not read memory: {e}")


def load_avg_metrics(results_dir: Path, model_name: str, method_name: str) -> dict:
    """Read the averaged metrics CSV for a completed model/method run."""
    csv_path = results_dir / f"{model_name}_{method_name}.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings back to floats
            return {
                k: float(v) if _is_float(v) else v
                for k, v in row.items()
            }
    return {}


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def main():
    with open("Experiment/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track all results for final comparison chart
    # Structure: {model_name: {method_name: avg_metrics_dict}}
    all_results = {}

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        methods    = model_cfg.get("methods", [])

        if not methods:
            print(f"\n[SKIP] {model_name} — no methods configured")
            continue

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        all_results[model_name] = {}

        for method_name in methods:
            print(f"\n--- Method: {method_name} ---")

            print("[MEMORY] Dropping caches before subprocess launch...")
            drop_caches()
            time.sleep(5)
            print(f"[MEMORY] Before {model_name}/{method_name}:")
            print_memory()

            cmd = [
                sys.executable,
                "run_method.py",
                "--model",  model_name,
                "--method", method_name
            ]

            result = subprocess.run(cmd, timeout=3600)

            print("[MEMORY] Dropping caches after subprocess exit...")
            drop_caches()
            time.sleep(5)
            print(f"[MEMORY] After {model_name}/{method_name}:")
            print_memory()

            if result.returncode == 0:
                print(f"[OK] {model_name}/{method_name} completed successfully")
                avg = load_avg_metrics(results_dir, model_name, method_name)
                all_results[model_name][method_name] = avg
            else:
                print(f"[ERROR] {model_name}/{method_name} failed "
                      f"with code {result.returncode} — skipping.")

    # -----------------------------------------------------------
    # Final cross-method comparison chart
    # -----------------------------------------------------------
    print("\n[GRAPHS] Generating cross-method comparison chart...")
    try:
        from Evaluation.graphs import plot_comparison
        comparison_path = results_dir / "comparison_all_models.png"
        plot_comparison(all_results, save_path=str(comparison_path))
        print(f"[SAVED] {comparison_path}")
    except Exception as e:
        print(f"[GRAPHS ERROR] Could not generate comparison chart: {e}")

    print("\n[DONE] All methods completed.")


if __name__ == "__main__":
    main()
