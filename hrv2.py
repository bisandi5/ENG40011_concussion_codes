import math


def calculate_variability(intervals_ms):
    """Calculate variability from consecutive pulse intervals."""

    if len(intervals_ms) < 3:
        raise ValueError("At least three pulse intervals are required.")

    if any(
        not math.isfinite(interval) or interval <= 0
        for interval in intervals_ms
    ):
        raise ValueError("Intervals must be positive, finite numbers.")

    mean_interval = sum(intervals_ms) / len(intervals_ms)

    differences = [
        intervals_ms[index + 1] - intervals_ms[index]
        for index in range(len(intervals_ms) - 1)
    ]

    rmssd = math.sqrt(
        sum(difference ** 2 for difference in differences)
        / len(differences)
    )

    return {
        "interval_count": len(intervals_ms),
        "mean_interval_ms": mean_interval,
        "pulse_rate_bpm": 60000 / mean_interval,
        "rmssd_ms": rmssd,
    }


if __name__ == "__main__":
    # Simulated intervals for checking the calculation.
    # These are not sensor measurements.
    demo_intervals = [800, 820, 790, 810, 800]

    result = calculate_variability(demo_intervals)

    print("SIMULATED DATA — calculation demonstration only")
    print(f"Intervals: {demo_intervals} ms")
    print(f"Mean interval: {result['mean_interval_ms']:.1f} ms")
    print(f"Pulse rate: {result['pulse_rate_bpm']:.1f} BPM")
    print(f"RMSSD: {result['rmssd_ms']:.2f} ms")