"""
Worker script -- runs a single model/method combination.
Called as a subprocess by experiment_runner.py.
Uses llama-cpp-python for inference on pre-prepared GGUF files.
When this script exits, ALL memory is returned to the OS --
guaranteed clean slate for the next method.

DPS Strategy: time-based precision switching.
Monitors seconds-per-token after each run. Since energy = power x time
and power is fixed on the Pi, s/token directly represents energy cost
per token. DPS switches to Q8_0 when s/token exceeds the efficiency
threshold, and back to fp16 when throughput recovers.

Threshold: 0.8s/token
  fp16 baseline: ~1.1s/token  (above threshold -> switches)
  Q8_0 baseline: ~0.59s/token (below threshold -> stays)
  On faster hardware fp16 may stay below threshold -> no switch needed.
"""

import argparse
import csv
import gc
import json
import time
import yaml
from pathlib import Path
from statistics import mean

from llama_cpp import Llama

from Evaluation.metrics import MetricsCollector, print_metrics_table
from Evaluation.graphs import plot_metrics


# DPS thresholds -- calibrated from measured Pi 400 performance
# Midpoint between fp16 (~1.1s/token) and Q8_0 (~0.59s/token)
DPS_SLOW_THRESHOLD = 0.80   # s/token -- above this: switch to Q8_0
DPS_FAST_THRESHOLD = 0.65   # s/token -- below this: switch back to fp16

# Models that use plain completion instead of chat completion
# because they echo the chat template instead of responding
PLAIN_COMPLETION_MODELS = ["llama3.2", "qwen2"]


def average_metrics(metrics_list):
    """Average numeric metrics only -- text fields use first run's value."""
    averaged = {}
    for key in metrics_list[0]:
        if key == "Response":
            continue  # excluded from summary -- saved in runs CSV only
        if isinstance(metrics_list[0][key], (int, float)):
            averaged[key] = round(mean(m[key] for m in metrics_list), 4)
        else:
            averaged[key] = metrics_list[0][key]
    return averaged


def load_model(gguf_path, n_ctx=512):
    print(f"[MODEL] Loading {gguf_path}...")
    model = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )
    print(f"[MODEL] Loaded successfully.")
    return model


def run_inference(model, prompt, max_new_tokens, use_plain_completion=False):
    """
    Run inference and return (output_text, token_count).
    Chat models (TinyLlama): use create_chat_completion for proper formatting.
    Base models (Llama3.2, Qwen2): use plain completion to avoid template echo.
    temperature=0.1 for near-deterministic consistent output.
    """
    if use_plain_completion:
        output = model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.1,
            top_p=1.0,
            echo=False
        )
        text   = output["choices"][0]["text"]
        tokens = output["usage"]["completion_tokens"]
    else:
        output = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.1,
            top_p=1.0,
        )
        text   = output["choices"][0]["message"]["content"]
        tokens = output["usage"]["completion_tokens"]

    return text, tokens


def get_spt(run_metrics):
    """Compute seconds-per-token. Returns None if tokens is 0."""
    t      = run_metrics.get("Inference Time (s)", 0)
    tokens = run_metrics.get("Tokens", 0)
    if tokens > 0:
        return round(t / tokens, 4)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    with open("Experiment/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results_dir     = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_per_method = config["runs_per_method"]
    max_new_tokens  = config["max_new_tokens"]
    gguf_dir        = config["gguf_dir"]

    model_cfg = next(
        (m for m in config["models"] if m["name"] == args.model), None
    )
    if model_cfg is None:
        print(f"[ERROR] Model {args.model} not found in config")
        exit(1)

    model_name  = model_cfg["name"]
    prompts     = model_cfg["prompts"]
    method_name = args.method

    # Determine whether to use plain completion for this model
    use_plain = model_name in PLAIN_COMPLETION_MODELS
    print(f"[WORKER] Running {model_name} / {method_name} "
          f"({'plain completion' if use_plain else 'chat completion'})")

    dps_mode = (method_name == "dps")

    if method_name == "fp16":
        gguf_path = f"{gguf_dir}/{model_name}_fp16.gguf"
    elif method_name == "int8":
        gguf_path = f"{gguf_dir}/{model_name}_q8.gguf"
    elif method_name == "pruned":
        gguf_path = f"{gguf_dir}/{model_name}_pruned.gguf"
    elif method_name == "dps":
        fp16_path = f"{gguf_dir}/{model_name}_fp16.gguf"
        q8_path   = f"{gguf_dir}/{model_name}_q8.gguf"
        gguf_path = fp16_path
        dps_current_precision = "fp16"
        dps_switch_log = []
    else:
        print(f"[ERROR] Unknown method: {method_name}")
        exit(1)

    if dps_mode:
        print(f"[DPS] Thresholds: slow={DPS_SLOW_THRESHOLD}s/token "
              f"fast={DPS_FAST_THRESHOLD}s/token")

    run_results = []
    responses    = []  # separate list for response CSV

    for i in range(runs_per_method):
        run_number = i + 1
        prompt     = prompts[i % len(prompts)]

        # DPS: evaluate previous run s/token and switch if needed
        if dps_mode and run_results:
            spt = get_spt(run_results[-1])

            if spt is not None:
                if spt > DPS_SLOW_THRESHOLD and dps_current_precision == "fp16":
                    print(f"[DPS] Run {run_number}: {spt:.3f}s/token > "
                          f"{DPS_SLOW_THRESHOLD}s/token -- "
                          f"too slow, switching fp16 -> Q8_0")
                    prev = dps_current_precision
                    dps_current_precision = "q8"
                    gguf_path = q8_path
                    dps_switch_log.append({
                        "switch_at_run":   run_number,
                        "from_precision":  prev,
                        "to_precision":    "q8",
                        "trigger_spt":     spt,
                        "threshold_spt":   DPS_SLOW_THRESHOLD,
                        "reason":          "throughput_too_slow"
                    })

                elif spt < DPS_FAST_THRESHOLD and dps_current_precision == "q8":
                    print(f"[DPS] Run {run_number}: {spt:.3f}s/token < "
                          f"{DPS_FAST_THRESHOLD}s/token -- "
                          f"fast enough, switching Q8_0 -> fp16")
                    prev = dps_current_precision
                    dps_current_precision = "fp16"
                    gguf_path = fp16_path
                    dps_switch_log.append({
                        "switch_at_run":   run_number,
                        "from_precision":  prev,
                        "to_precision":    "fp16",
                        "trigger_spt":     spt,
                        "threshold_spt":   DPS_FAST_THRESHOLD,
                        "reason":          "throughput_recovered"
                    })

                else:
                    print(f"[DPS] Run {run_number}: {spt:.3f}s/token -- "
                          f"staying on {dps_current_precision}")
            else:
                print(f"[DPS] Run {run_number}: no s/token reading -- "
                      f"staying on {dps_current_precision}")

        elif dps_mode:
            print(f"[DPS] Run {run_number}: first run -- "
                  f"starting on {dps_current_precision} (full precision)")

        # Load fresh model each run -- clean context window every time
        model = load_model(gguf_path)

        with MetricsCollector(method_name=method_name, model_name=model_name,
                              energy_logger=None) as metrics:
            metrics.start()
            output_text, token_count = run_inference(
                model, prompt, max_new_tokens, use_plain_completion=use_plain
            )
            metrics.stop()
            run_metrics = metrics.compute_metrics(token_count)
            if dps_mode:
                spt = get_spt(run_metrics)
                run_metrics["dps_precision"] = dps_current_precision
                run_metrics["s_per_token"]   = spt if spt is not None else 0.0
            run_results.append(run_metrics)
            responses.append({
                "Run":      run_number,
                "Prompt":   prompt,
                "Response": output_text.strip()
            })

        del model
        gc.collect()

        spt_str = ""
        if dps_mode:
            spt = get_spt(run_metrics)
            spt_str = f" | {spt:.3f}s/tok [{dps_current_precision}]" if spt else ""

        print(f"[RUN {run_number}/{runs_per_method}] "
              f"{token_count} tokens | "
              f"{run_metrics.get('Inference Time (s)', 0):.1f}s | "
              f"{run_metrics.get('Avg Power (W)', 0):.2f}W | "
              f"{run_metrics.get('Joule / Token', 0):.3f} J/token"
              + spt_str)
        preview = output_text[:150].replace("\n", " ").strip()
        print(f"  >> {preview}...")

    # Aggregate & Save
    avg_metrics   = average_metrics(run_results)  # Response excluded here
    csv_path      = results_dir / f"{model_name}_{method_name}.csv"
    fig_path      = results_dir / f"{model_name}_{method_name}.png"

    # Summary CSV -- numbers only, no Response column
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=avg_metrics.keys())
        writer.writeheader()
        writer.writerow(avg_metrics)

    # Responses CSV -- numbered list of responses only
    responses_csv_path = results_dir / f"{model_name}_{method_name}_responses.csv"
    with open(responses_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in responses:
            writer.writerow([f"{r['Run']}. {r['Response']}"])

    if dps_mode:
        switch_log_path = results_dir / f"{model_name}_dps_switch_log.json"
        with open(switch_log_path, "w") as f:
            json.dump({
                "model":              model_name,
                "total_runs":         runs_per_method,
                "slow_threshold_spt": DPS_SLOW_THRESHOLD,
                "fast_threshold_spt": DPS_FAST_THRESHOLD,
                "switches":           dps_switch_log,
                "total_switches":     len(dps_switch_log),
                "final_precision":    dps_current_precision,
            }, f, indent=2)
        print(f"[DPS] {len(dps_switch_log)} switch(es) recorded.")
        print(f"[SAVED] {switch_log_path}")

    print_metrics_table([avg_metrics])

    plot_metrics(
        metrics_list=run_results,
        title=f"{model_name} - {method_name}",
        save_path=str(fig_path)
    )

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {responses_csv_path}")
    print(f"[SAVED] {fig_path}")
    print(f"[WORKER] {model_name}/{method_name} done -- exiting cleanly")


if __name__ == "__main__":
    main()
