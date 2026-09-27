import time
import threading
from typing import Dict, Any

from Monitoring.energy_logger import EnergyLogger
from Monitoring.perfomance_monitor import (
    get_cpu_usage,
    get_ram_usage
)


class MetricsCollector:
    def __init__(
        self,
        method_name: str,
        model_name: str,
        energy_logger: EnergyLogger = None
    ):
        self.method_name = method_name
        self.model_name = model_name

        self._owns_energy_logger = energy_logger is None
        self.energy_logger = energy_logger if energy_logger is not None else EnergyLogger()

        self._stop_event = threading.Event()
        self._thread = None

        self.power_samples = []
        self.cpu_samples = []
        self.ram_samples = []
        self.timestamps = []

        self.start_time = None
        self.end_time = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                data = self.energy_logger.read()

                # Check again after blocking read —
                # stop_event may have been set while we were waiting
                if self._stop_event.is_set():
                    break

                self.power_samples.append(data.power_W)
                self.cpu_samples.append(get_cpu_usage())
                self.ram_samples.append(get_ram_usage())
                self.timestamps.append(time.time())

            except Exception as e:
                print(f"[METRICS WARNING] {e}")

    def start(self):
        self.power_samples.clear()
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.timestamps.clear()

        self._stop_event.clear()
        self.start_time = time.time()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # Signal the thread to stop
        self._stop_event.set()

        if self._thread is not None:
            # Wait up to 5 seconds for the thread to finish its current read()
            self._thread.join(timeout=5)
            self._thread = None

        self.end_time = time.time()

    def close(self):
        if self._owns_energy_logger:
            self.energy_logger.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def compute_metrics(self, generated_tokens: int) -> Dict[str, Any]:
        duration = self.end_time - self.start_time

        if len(self.power_samples) == 0:
            raise RuntimeError("No power samples collected")

        if len(self.power_samples) == 1:
            energy_joules = self.power_samples[0] * duration
        else:
            energy_joules = sum(
                ((self.power_samples[i] + self.power_samples[i - 1]) / 2)
                * (self.timestamps[i] - self.timestamps[i - 1])
                for i in range(1, len(self.power_samples))
            )

        avg_power = sum(self.power_samples) / len(self.power_samples)
        avg_cpu   = sum(self.cpu_samples)   / len(self.cpu_samples)
        avg_ram   = sum(self.ram_samples)   / len(self.ram_samples)

        joule_per_token = (
            energy_joules / generated_tokens
            if generated_tokens > 0 else 0.0
        )

        return {
            "Method":             self.method_name,
            "Model":              self.model_name,
            "Inference Time (s)": round(duration, 3),
            "Avg Power (W)":      round(avg_power, 3),
            "Total Energy (J)":   round(energy_joules, 3),
            "Joule / Token":      round(joule_per_token, 4),
            "Avg CPU (%)":        round(avg_cpu, 2),
            "Avg RAM (%)":        round(avg_ram, 2),
            "Tokens":             generated_tokens,
            "Power Samples":      len(self.power_samples)
        }


def print_metrics_table(results: list):
    headers = results[0].keys()
    col_widths = {h: max(len(h), 12) for h in headers}

    for row in results:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    def print_row(row_dict):
        return " | ".join(
            str(row_dict[h]).ljust(col_widths[h]) for h in headers
        )

    print("\n" + "=" * 100)
    print(print_row({h: h for h in headers}))
    print("-" * 100)

    for r in results:
        print(print_row(r))

    print("=" * 100 + "\n")