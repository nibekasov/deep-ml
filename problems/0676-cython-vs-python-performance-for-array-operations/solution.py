import math


def analyze_array_performance(n: int, operations: list) -> dict:
    methods = ("python", "cython", "numpy")
    totals = dict.fromkeys(methods, 0.0)
    results = []
    crossovers = {}

    for op in operations:
        times = {
            m: (
                op[f"{m}_overhead_ns"]
                + n * op[f"{m}_per_elem_ns"]
            ) / 1000
            for m in methods
        }

        # Накапливаем до округления.
        for m in methods:
            totals[m] += times[m]

        results.append({
            "name": op["name"],
            **{
                f"{m}_us": round(times[m], 2)
                for m in methods
            },
            "cython_speedup": round(
                times["python"] / times["cython"], 2
            ),
            "numpy_speedup": round(
                times["python"] / times["numpy"], 2
            ),
            "best": min(
                methods, key=lambda m: (times[m], m)
            ),
        })

        delta_pe = (
            op["cython_per_elem_ns"]
            - op["numpy_per_elem_ns"]
        )
        delta_oh = (
            op["numpy_overhead_ns"]
            - op["cython_overhead_ns"]
        )

        crossovers[op["name"]] = (
            -1 if delta_pe <= 0
            else max(0, math.ceil(delta_oh / delta_pe))
        )

    return {
        "operations": results,
        **{
            f"pipeline_{m}_us": round(totals[m], 2)
            for m in methods
        },
        "recommended": min(
            methods, key=lambda m: (totals[m], m)
        ),
        "crossover_sizes": crossovers,
    }