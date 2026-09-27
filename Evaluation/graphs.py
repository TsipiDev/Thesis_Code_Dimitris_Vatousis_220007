import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless Pi use
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics_list: list, title: str, save_path: str = None):
    """
    Per-method run chart — shows how each metric evolves across
    the 5 runs within a single method. Useful for spotting
    variance and DPS switching effects.
    """
    runs = [f"Run {i+1}" for i in range(len(metrics_list))]

    inference_time  = [r["Inference Time (s)"] for r in metrics_list]
    avg_power       = [r["Avg Power (W)"]       for r in metrics_list]
    total_energy    = [r["Total Energy (J)"]    for r in metrics_list]
    joule_per_token = [r["Joule / Token"]       for r in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Colour runs by DPS precision if available
    colors = []
    for r in metrics_list:
        if r.get("dps_precision") == "q8":
            colors.append("darkorange")
        elif r.get("dps_precision") == "fp16":
            colors.append("steelblue")
        else:
            colors.append("steelblue")

    def bar(ax, values, ylabel, color_override=None):
        c = color_override if color_override else colors
        ax.bar(runs, values, color=c)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for idx, v in enumerate(values):
            ax.text(idx, v * 1.01, f"{v:.2f}", ha="center", fontsize=8)

    bar(axes[0, 0], inference_time,  "Seconds")
    axes[0, 0].set_title("Inference Time (s)")

    bar(axes[0, 1], avg_power, "Watts")
    axes[0, 1].set_title("Average Power (W)")

    bar(axes[1, 0], total_energy, "Joules", color_override="green")
    axes[1, 0].set_title("Total Energy (J)")

    bar(axes[1, 1], joule_per_token, "J / token", color_override="crimson")
    axes[1, 1].set_title("Energy Efficiency (J / Token)")

    # DPS legend
    if any(r.get("dps_precision") for r in metrics_list):
        from matplotlib.patches import Patch
        legend = [
            Patch(color="steelblue",  label="fp16"),
            Patch(color="darkorange", label="Q8_0")
        ]
        fig.legend(handles=legend, loc="lower center", ncol=2,
                   title="DPS Precision", bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAPH] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_comparison(all_results: dict, save_path: str = None):
    """
    Cross-method comparison chart — the main thesis chart.
    all_results = {
        "tinyllama": {"fp16": avg_metrics, "int8": avg_metrics, ...},
        "llama3.2":  {...},
        "qwen2":     {...}
    }
    Shows fp16 / int8 / pruned / dps grouped by model for each metric.
    """
    methods = ["fp16", "int8", "pruned", "dps"]
    models  = list(all_results.keys())

    method_colors = {
        "fp16":   "steelblue",
        "int8":   "darkorange",
        "pruned": "mediumseagreen",
        "dps":    "mediumpurple"
    }

    metrics_to_plot = [
        ("Avg Power (W)",      "Average Power (W)",        "Watts"),
        ("Total Energy (J)",   "Total Energy (J)",          "Joules"),
        ("Joule / Token",      "Energy Efficiency (J/Token)", "J / token"),
        ("Inference Time (s)", "Inference Time (s)",        "Seconds"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Method Comparison Across Models", fontsize=15, fontweight="bold")

    x      = np.arange(len(models))
    n      = len(methods)
    width  = 0.18
    offset = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)

    for ax, (metric_key, metric_title, ylabel) in zip(axes.flat, metrics_to_plot):
        for j, method in enumerate(methods):
            values = []
            for model in models:
                v = all_results.get(model, {}).get(method, {}).get(metric_key, 0)
                values.append(v)

            bars = ax.bar(
                x + offset[j], values, width,
                label=method,
                color=method_colors[method],
                alpha=0.85
            )

            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7
                    )

        ax.set_title(metric_title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAPH] Comparison chart saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
