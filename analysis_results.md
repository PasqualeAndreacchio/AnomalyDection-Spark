# Benchmark Results Analysis — Critical Issues Found

## The Core Problem: Repeat 2 is a Cache Hit, Not a Real Measurement

Looking at the raw data in [benchmark_results.json](file:///home/matteocalcagni/Desktop/MAPD_B/project/AnomalyDection-Spark/Systematic_Distributed_Benchmarking/benchmark_results.json), repeat 2 for **every task at every core count** has suspiciously tiny values. Here's the evidence:

### Task 5A: Switch Counts (1 core)

| Repeat | Wall Time (s) | CPU Time (s) | Shuffle Read (bytes) |
|--------|--------------|-------------|---------------------|
| 1      | **604.35**   | 408.93      | 1,036,754,832       |
| 2      | **0.43**     | 0.04        | 708                 |
| 3      | **582.87**   | 398.51      | 1,036,754,832       |
| 4      | **589.15**   | 402.03      | 1,036,754,832       |

> [!CAUTION]
> Repeat 2 takes **0.43 seconds** instead of ~590 seconds. It reads **708 bytes** instead of ~1 GB. This is not a real computation — Spark is returning the cached result from Repeat 1 instantly.

This happens identically for **every task at every core count** (Tasks 5A, 5B, 5C × 1, 2, 4, 6 cores).

## Why This Happens

Looking at the benchmark loop in [benchmark_suite.py](file:///home/matteocalcagni/Desktop/MAPD_B/project/AnomalyDection-Spark/Systematic_Distributed_Benchmarking/benchmark_suite.py#L443-L488):

```python
for r in range(args.repeats):  # repeats = 4
    # ...
    df_anomaly1, df_hourly_frequency = run_switch_counts(df_aggregated, target_metrics)
    
    df_anomaly1.cache()
    df_hourly_frequency.cache()
    res = df_hourly_frequency.count()   # <-- materialise
    
    # Clean up cache from previous repeats
    if r > 0 and df_anomaly1_saved is not None:
        df_anomaly1_saved.unpersist()    # <-- unpersist AFTER caching the new one
        df_hourly_freq_saved.unpersist()
    
    df_anomaly1_saved = df_anomaly1
    df_hourly_freq_saved = df_hourly_frequency
```

The problem is a **deterministic Spark DAG deduplication** issue:

1. **Repeat 1**: `run_switch_counts(df_aggregated, ...)` builds a Spark DAG. `.cache().count()` materialises it and stores the result in memory. Wall time = ~600s. ✅
2. **Repeat 2**: `run_switch_counts(df_aggregated, ...)` builds the **exact same DAG** again (same input, same transformations, same lineage). When `.cache().count()` is called, Spark recognizes this is the same logical plan and **serves the result directly from the existing cache**. Wall time = ~0.4s. ❌ This is a cache hit, not real work.
3. **Repeat 3**: After repeat 2, the code runs `df_anomaly1_saved.unpersist()` which removes the cache from repeat 2 (which was actually the cache from repeat 1). Now the cache is gone, so repeat 3 has to recompute. Wall time = ~580s. ✅
4. **Repeat 4**: Same as repeat 3 — previous cache was unpersisted, so real work happens. Wall time = ~590s. ✅

> [!IMPORTANT]
> **Every other repeat (2nd) is a cache hit, not genuine computation.** The `unpersist()` of the previous repeat only happens AFTER the new repeat has already cached and materialised, so repeat 2 always piggybacks on repeat 1's cache.

## How This Corrupts Your Results

The summary computes `mean` across **all 4 repeats**, including the fake ~0.4s repeat 2:

**Task 5A, 1 core:**
- True mean of repeats 1, 3, 4: `(604 + 583 + 589) / 3 ≈ 592s`
- Reported mean (including repeat 2): `(604 + 0.4 + 583 + 589) / 4 ≈ 444s`

**This inflates speedup at 1 core** (the baseline), making the baseline appear faster than it truly is.

The same bug hits **every core count**, but the impact is proportional to the absolute task duration. For the 1-core Task 5A baseline (~600s), including a ~0.4s outlier pulls the mean down by ~150s. For the 6-core run (~130s), including a ~0.2s outlier only pulls it down by ~33s. This asymmetric distortion means:
- The 1-core baseline is artificially **too fast** → Speedup is **underestimated**
- The standard deviations are **massively inflated** (e.g., Task 5A 1-core std = 296s, which is clearly dominated by the outlier)

## Impact on Efficiency Values

Your efficiency values above 1.0 at 2 and 4 cores, and then the sudden drop at 6 cores, are **partially an artifact of this bug**. The corrupted 1-core baseline makes the reference T(1) unreliable, so all derived speedup and efficiency calculations are distorted.

## The Fix

Move the `unpersist()` call to **before** the new computation starts, so the cache is cleared before Spark builds the new DAG:

```diff
 for r in range(args.repeats):
+    # Clean up cache from previous repeat BEFORE new computation
+    if r > 0 and df_anomaly1_saved is not None:
+        df_anomaly1_saved.unpersist()
+        df_hourly_freq_saved.unpersist()
+
     start_stage = get_current_max_stage(app_id)
     t0 = time.perf_counter()
     
     df_anomaly1, df_hourly_frequency = run_switch_counts(df_aggregated, target_metrics)
     
     df_anomaly1.cache()
     df_hourly_frequency.cache()
     res = df_hourly_frequency.count()
     
     t1 = time.perf_counter()
     wall_time = t1 - t0
     
-    # Clean up cache from previous repeats to avoid memory leaks
-    if r > 0 and df_anomaly1_saved is not None:
-        df_anomaly1_saved.unpersist()
-        df_hourly_freq_saved.unpersist()
-    
     df_anomaly1_saved = df_anomaly1
     df_hourly_freq_saved = df_hourly_frequency
```

The same fix must be applied to the **Task 5B** loop as well (line ~512-514).

> [!NOTE]
> This ensures every repeat actually recomputes from scratch, giving you valid, consistent wall-clock times across all repeats.
