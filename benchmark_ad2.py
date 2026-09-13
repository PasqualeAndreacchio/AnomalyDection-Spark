#!/usr/bin/env python3
"""
CloudVeneto Spark Distributed Benchmark — Anomaly Detection 2
Group 5, MAPD Part B.

Adapted from the benchmark suite written by Nicola Lavarda for Task 3.3.3.
Structure, metric collection and plotting are his; the three benchmarked
workloads are replaced with the ones from the Anomaly Detection 2 pipeline
(device load vs external temperature).

The three phases follow the ones the assignment asks to characterise:

  1. MAP-LIKE      ingestion and translation into the working structure: read,
                   filter three metrics out of ~120, apply the decimal scaling.
                   Narrow only, no shuffle.
  2. GROUP-LIKE    the statistical core: minute resampling, long->wide pivot,
                   hourly aggregation and the per-device correlations. Four
                   wide dependencies, one after the other.
  3. ADVANCED MIX  the analysis that yields the findings: binned temperature
                   profile, duty cycle vs load while running, and a global
                   ordering with no partitionBy. Different keys, repeated
                   reshuffling, and one stage that cannot parallelise at all.

If phase 1 scaled badly that would be a warning sign, since nothing in it
requires the nodes to talk to each other.

A measured speed-up above the ideal line is not a result, it is an artefact:
usually it means the low-core runs were timed on a cold system and the
high-core ones on a warm one. Each session therefore runs a full warm-up that
is discarded, and every configuration is repeated so the median absorbs the
remaining noise.

HOW TO RUN (from the master, with the venv active and no notebook kernel
holding a Spark session):

    source /opt/mapd-project/pyvenv/bin/activate
    export S3_ACCESS_KEY=...        # already in .bashrc, harmless to repeat
    export S3_PRIVATE_KEY=...
    nohup python3 benchmark_ad2.py --output-dir ~/benchmark_ad2 \
          > benchmark.log 2>&1 &
    tail -f benchmark.log
"""

import os
import time
import json
import argparse
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
from pyspark.sql import Window

# seaborn is optional: the script falls back to plain matplotlib if missing
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("seaborn not installed, using plain matplotlib styling", flush=True)


DATA_PATH = "s3a://MAPDB-Group5/data_parquet"
LOAD_COLS = ["S125", "S181"]
TEMP_COL = "S41"
TARGET = LOAD_COLS + [TEMP_COL]


# ==============================================================================
# SESSION AND METRICS
# ==============================================================================

def get_spark_session(app_name="AD2_Benchmark", cores_max=6, shuffle_partitions=12,
                      executor_memory="1g"):
    """A session with an explicit core cap. Standalone manager, so the cap is
    spark.cores.max: --num-executors is a YARN flag and would be ignored.

    Note what raising cores_max does on this cluster. Each worker starts one
    executor with `executor_memory` of heap, so asking for 1 core involves one
    worker and 1 x executor_memory, while asking for 6 involves all three and
    3 x executor_memory. Cores and memory therefore grow together, and a
    low-core run that spills to disk will look disproportionately slow. That is
    a property of the cluster, not of the pipeline, and it is why the memory is
    kept modest here."""
    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_PRIVATE_KEY") or os.getenv("S3_SECRET_KEY")
    if not access_key or not secret_key:
        raise SystemExit("S3 credentials not found in the environment.")

    return (
        SparkSession.builder
        .master("spark://master:7077")
        .appName(f"{app_name}_Cores_{cores_max}")
        .config("spark.cores.max", str(cores_max))
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", "1536m")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.4.1,org.apache.hadoop:hadoop-common:3.4.1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.endpoint", "https://cloud-areapd.pd.infn.it:5210")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )


def _query_stages(app_id, ui_port):
    url = f"http://localhost:{ui_port}/api/v1/applications/{app_id}/stages"
    with urllib.request.urlopen(url, timeout=5) as res:
        return json.loads(res.read().decode())


def get_current_max_stage(app_id, ui_port):
    try:
        stages = _query_stages(app_id, ui_port)
        return max((s.get("stageId", 0) for s in stages), default=0)
    except Exception:
        return 0


def get_stage_metrics(app_id, ui_port, min_stage_id=0):
    """Stage-level counters from the Spark REST API. Returns zeros if the UI
    is unreachable, so a missing metric never aborts the run."""
    empty = {"executorRunTime_sec": 0.0, "executorCpuTime_sec": 0.0,
             "jvmGcTime_sec": 0.0, "shuffleReadBytes": 0, "shuffleWriteBytes": 0,
             "numTasks": 0}
    try:
        stages = _query_stages(app_id, ui_port)
    except Exception as e:
        print(f"    warning: Spark REST API unreachable ({e})", flush=True)
        return empty

    agg = dict(empty)
    for s in stages:
        if s.get("stageId", 0) <= min_stage_id:
            continue
        agg["executorRunTime_sec"] += s.get("executorRunTime", 0) / 1000.0
        agg["executorCpuTime_sec"] += s.get("executorCpuTime", 0) / 1e9
        agg["jvmGcTime_sec"] += s.get("jvmGcTime", 0) / 1000.0
        agg["shuffleReadBytes"] += s.get("shuffleReadBytes", 0)
        agg["shuffleWriteBytes"] += s.get("shuffleWriteBytes", 0)
        agg["numTasks"] += s.get("numTasks", 0)
    return agg


# ==============================================================================
# THE THREE WORKLOADS
# ==============================================================================

def run_map_bound(spark):
    """Narrow only: read, filter three metrics out of ~120, apply the decimal
    scaling. No shuffle, so this should scale close to linearly until I/O
    becomes the bottleneck."""
    df = spark.read.parquet(DATA_PATH).where(sf.col("metric").isin(TARGET))
    df = df.withColumn(
        "value_scaled",
        sf.when(sf.col("metric") == TEMP_COL, sf.col("value") / 10.0)
          .otherwise(sf.col("value").cast("double"))
    )
    return df.count()


def run_group_like(spark):
    """Phase 2: the statistical core of the analysis.

    Minute resampling, long->wide pivot, hourly aggregation, and the per-device
    Pearson correlations. Everything here is a wide dependency: four shuffles,
    each redistributing rows so that those sharing a key end up together.

    The correlations are computed at minute level as well as hourly. At hourly
    level the table is only ~13k rows, small enough that scheduling overhead
    would dominate the measurement and hide whatever the correlation itself
    costs; at minute level there are ~670k rows, enough for the work to be
    worth distributing.
    """
    df = (spark.read.parquet(DATA_PATH)
               .where(sf.col("metric").isin(TARGET))
               .withColumn("timestamp_bucket",
                           sf.date_trunc("minute", sf.timestamp_millis(sf.col("when")))))

    # shuffle 1: resample to one row per (device, minute, metric)
    df_res = (df.groupBy("timestamp_bucket", "hwid", "metric")
                .agg(sf.mean("value").alias("value")))

    # shuffle 2: pivot long -> wide
    df_wide = (df_res.groupBy("timestamp_bucket", "hwid")
                     .pivot("metric", TARGET)
                     .agg(sf.first("value"))
                     .withColumn(TEMP_COL, sf.col(TEMP_COL) / 10.0))

    n_present = sum(sf.when(sf.col(c).isNotNull(), 1).otherwise(0) for c in LOAD_COLS)
    sum_load = sum(sf.coalesce(sf.col(c), sf.lit(0.0)) for c in LOAD_COLS)
    df_wide = (df_wide
               .withColumn("load_avg", sf.when(n_present > 0, sum_load / n_present))
               .withColumn("hour_bin", sf.date_trunc("hour", sf.col("timestamp_bucket")))
               .cache())
    n = df_wide.count()

    # shuffle 3: per-device correlations on the full minute-level table
    n += (df_wide.groupBy("hwid")
                 .agg(*[sf.corr(c, TEMP_COL).alias(f"corr_{c}")
                        for c in LOAD_COLS + ["load_avg"]])
                 .count())

    # shuffle 4: hourly aggregation, then the same correlations on it
    df_hourly = (df_wide.groupBy("hwid", "hour_bin")
                        .agg(*[sf.mean(c).alias(c) for c in TARGET],
                             sf.mean("load_avg").alias("load_avg")))
    n += (df_hourly.groupBy("hwid")
                   .agg(sf.corr("load_avg", TEMP_COL).alias("corr_hourly"))
                   .count())

    return n, df_wide


def run_advanced_mix(spark, df_wide):
    """Phase 3: the analysis that produces the actual findings.

    Three computations on different keys, coordinated: the binned temperature
    profile, the split between duty cycle and load while running, and a global
    ordering. Each groups on a different key, so the data is reshuffled each
    time, and the last one has no partitionBy at all — Spark funnels every row
    into a single partition there, so that part cannot use more than one core
    however many are allocated.

    This mixture is deliberate: it is the shape the real analysis has, and its
    scaling curve should sit below the purely group-like phase.
    """
    df_ok = df_wide.where(sf.col(TEMP_COL).isNotNull() & sf.col("load_avg").isNotNull())

    # binned temperature profile, grouped on (device, temperature bin)
    n = (df_ok.withColumn("temp_bin", sf.floor(sf.col(TEMP_COL) / 2.0) * 2.0)
              .groupBy("hwid", "temp_bin")
              .agg(sf.mean("load_avg").alias("mean_load"),
                   sf.stddev("load_avg").alias("std_load"),
                   sf.count("*").alias("n"))
              .count())

    # duty cycle vs load while running, same key, different aggregations
    n += (df_ok.withColumn("temp_bin", sf.floor(sf.col(TEMP_COL) / 2.0) * 2.0)
               .groupBy("hwid", "temp_bin")
               .agg(sf.mean(sf.when(sf.col("load_avg") > 0, 1.0).otherwise(0.0))
                      .alias("active_fraction"),
                    sf.mean(sf.when(sf.col("load_avg") > 0, sf.col("load_avg")))
                      .alias("load_running"))
               .count())

    # global ordering: no partitionBy, so a single partition by construction
    df_hourly = (df_ok.groupBy("hwid", "hour_bin")
                      .agg(sf.mean("load_avg").alias("load_avg"))
                      .cache())
    df_hourly.count()
    n += (df_hourly.withColumn("rk", sf.percent_rank().over(Window.orderBy("load_avg")))
                   .count())
    df_hourly.unpersist()

    return n


WORKLOADS = [
    ("1. Map-like (ingestion + rescale)",   "Map / Narrow"),
    ("2. Group-like (aggregate + correlate)", "Group / Wide"),
    ("3. Advanced mix (profile + window)",  "Mixed"),
]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="AD2 Spark scaling benchmark")
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--shuffle-test", action="store_true", default=True)
    parser.add_argument("--ui-port", type=int, default=4040,
                        help="Spark UI port; raise it if 4040 is already taken")
    parser.add_argument("--executor-memory", type=str, default="1g",
                        help="heap per executor; kept small so that raising the "
                             "core count does not also multiply the memory")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.expanduser("~/benchmark_ad2"))
    args = parser.parse_args()

    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "benchmark_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 78)
    print("SPARK SCALING BENCHMARK — ANOMALY DETECTION 2")
    print(f"data       : {DATA_PATH}")
    print(f"cores      : {args.cores}")
    print(f"repeats    : {args.repeats}")
    print(f"output     : {output_dir}")
    print("=" * 78, flush=True)

    records = []

    for n_cores in args.cores:
        print(f"\n>>> session with spark.cores.max = {n_cores}", flush=True)
        spark = get_spark_session(cores_max=n_cores,
                                  executor_memory=args.executor_memory)
        app_id = spark.sparkContext.applicationId
        print(f"    app id: {app_id}")
        print(f"    defaultParallelism: {spark.sparkContext.defaultParallelism}")

        # Warm-up, discarded. A limit(100) would not be enough: the core
        # configurations run in order 1, 2, 4, 6, so the first session pays
        # for cold caches, S3 connection setup and JVM JIT while the last one
        # finds everything warm. Left uncorrected that alone can push the
        # measured speed-up above the ideal line, which is impossible.
        # Running the full map-bound workload once per session and throwing
        # the result away puts every configuration on the same footing.
        try:
            t_warm = time.perf_counter()
            run_map_bound(spark)
            print(f"    warm-up (discarded): {time.perf_counter() - t_warm:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"    warm-up failed: {e}", flush=True)

        df_wide_cached = None

        for r in range(args.repeats):
            for label, category in WORKLOADS:
                s0 = get_current_max_stage(app_id, args.ui_port)
                t0 = time.perf_counter()

                if label.startswith("1."):
                    res = run_map_bound(spark)
                elif label.startswith("2."):
                    res, df_wide_cached = run_group_like(spark)
                else:
                    if df_wide_cached is None:
                        _, df_wide_cached = run_group_like(spark)
                    res = run_advanced_mix(spark, df_wide_cached)

                wall = time.perf_counter() - t0
                m = get_stage_metrics(app_id, args.ui_port, min_stage_id=s0)

                # one executor per worker, so the number of workers involved
                # is roughly ceil(cores / cores_per_worker)
                n_exec = max(1, min(3, -(-n_cores // 2)))
                records.append({
                    "task_name": label, "task_category": category,
                    "cores": n_cores, "repeat": r + 1,
                    "executors": n_exec,
                    "total_exec_memory_gb": n_exec * float(
                        args.executor_memory.rstrip("gG")),
                    "wall_time_sec": wall,
                    "cpu_time_sec": m["executorCpuTime_sec"],
                    "gc_time_sec": m["jvmGcTime_sec"],
                    "shuffle_read_bytes": m["shuffleReadBytes"],
                    "shuffle_write_bytes": m["shuffleWriteBytes"],
                    "num_tasks": m["numTasks"],
                    "result_count": res,
                })
                print(f"    [{label[:28]:28s} | cores {n_cores} | rep {r+1}] "
                      f"wall {wall:7.2f}s  cpu {m['executorCpuTime_sec']:7.2f}s  "
                      f"gc {m['jvmGcTime_sec']:5.2f}s  tasks {m['numTasks']}",
                      flush=True)

            if df_wide_cached is not None:
                df_wide_cached.unpersist()
                df_wide_cached = None

        spark.stop()
        time.sleep(2)

    # --------------------------------------------------------------------
    # shuffle partitions tuning, at the largest core count
    # --------------------------------------------------------------------
    shuffle_records = []
    if args.shuffle_test:
        max_cores = max(args.cores)
        print("\n" + "=" * 78)
        print(f"SHUFFLE PARTITIONS TUNING (cores = {max_cores})")
        print("=" * 78, flush=True)

        for p in [6, 12, 32, 200]:
            print(f"\n>>> spark.sql.shuffle.partitions = {p}", flush=True)
            spark = get_spark_session(cores_max=max_cores, shuffle_partitions=p,
                                      executor_memory=args.executor_memory)
            app_id = spark.sparkContext.applicationId
            s0 = get_current_max_stage(app_id, args.ui_port)
            t0 = time.perf_counter()
            _, dfw = run_group_like(spark)
            wall = time.perf_counter() - t0
            m = get_stage_metrics(app_id, args.ui_port, min_stage_id=s0)
            dfw.unpersist()

            shuffle_records.append({
                "shuffle_partitions": p, "wall_time_sec": wall,
                "cpu_time_sec": m["executorCpuTime_sec"],
                "gc_time_sec": m["jvmGcTime_sec"],
                "shuffle_read_bytes": m["shuffleReadBytes"],
                "num_tasks": m["numTasks"],
            })
            print(f"    partitions {p:3d}: wall {wall:6.2f}s  tasks {m['numTasks']}  "
                  f"shuffle read {m['shuffleReadBytes']/(1024**2):.1f} MB", flush=True)
            spark.stop()
            time.sleep(2)

    # --------------------------------------------------------------------
    # aggregation, speedup, efficiency
    # --------------------------------------------------------------------
    df_res = pd.DataFrame(records)
    summary = (df_res.groupby(["task_name", "task_category", "cores"])
                     .agg(wall_time_mean=("wall_time_sec", "mean"),
                          executors=("executors", "first"),
                          total_exec_memory_gb=("total_exec_memory_gb", "first"),
                          wall_time_std=("wall_time_sec", "std"),
                          cpu_time_mean=("cpu_time_sec", "mean"),
                          gc_time_mean=("gc_time_sec", "mean"),
                          shuffle_read_mean=("shuffle_read_bytes", "mean"),
                          num_tasks_mean=("num_tasks", "mean"))
                     .reset_index())

    base = summary[summary["cores"] == min(args.cores)] \
        .set_index("task_name")["wall_time_mean"].to_dict()
    summary["speedup"] = summary.apply(
        lambda r: base.get(r["task_name"], r["wall_time_mean"]) / r["wall_time_mean"]
        if r["wall_time_mean"] > 0 else 1.0, axis=1)
    summary["efficiency"] = summary["speedup"] / (summary["cores"] / min(args.cores))

    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary.to_csv(csv_path, index=False)
    df_res.to_csv(os.path.join(output_dir, "benchmark_raw.csv"), index=False)
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump({"raw": records, "summary": summary.to_dict("records"),
                   "shuffle_tuning": shuffle_records}, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(summary[["task_name", "cores", "executors", "total_exec_memory_gb",
                   "wall_time_mean", "speedup", "efficiency"]].to_string(index=False))
    print(f"\nsaved: {csv_path}")

    over = summary[summary["efficiency"] > 1.05]
    if not over.empty:
        print("\n" + "!" * 70)
        print("Efficiency above 1 is not a result, it is an artefact.")
        print("On this cluster more cores also means more executors and more")
        print("memory, so a low-core baseline that spills to disk makes every")
        print("other configuration look better than it is. Report the speed-up")
        print("alongside the memory column rather than on its own.")
        print("!" * 70)
        print(over[["task_name", "cores", "total_exec_memory_gb",
                    "efficiency"]].to_string(index=False))

    # --------------------------------------------------------------------
    # plots
    # --------------------------------------------------------------------
    if HAS_SNS:
        sns.set_theme(style="whitegrid", font_scale=1.05)

    tasks = summary["task_name"].unique()
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(tasks), 3)))

    # speedup
    plt.figure(figsize=(9, 5.5))
    cores_arr = np.array(sorted(args.cores))
    plt.plot(cores_arr, cores_arr / min(args.cores), "k--",
             label="ideal linear", linewidth=1.4, alpha=0.7)
    for i, t in enumerate(tasks):
        s = summary[summary["task_name"] == t].sort_values("cores")
        plt.plot(s["cores"], s["speedup"], marker="o", linewidth=2, label=t,
                 color=colours[i])
    plt.xlabel("total executor cores")
    plt.ylabel(f"speed-up vs {min(args.cores)} core(s)")
    plt.title("Strong scaling — Anomaly Detection 2 workloads\n"
              f"(one executor per worker, {args.executor_memory} each: "
              "cores and memory grow together)", fontsize=11)
    plt.xticks(sorted(args.cores))
    plt.legend()
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "plot1_speedup.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"plot: {p1}")

    # efficiency
    plt.figure(figsize=(9, 5.5))
    plt.axhline(100, color="k", linestyle="--", alpha=0.7, label="ideal 100%")
    for i, t in enumerate(tasks):
        s = summary[summary["task_name"] == t].sort_values("cores")
        plt.plot(s["cores"], s["efficiency"] * 100, marker="s", linewidth=2,
                 label=t, color=colours[i])
    plt.xlabel("total executor cores")
    plt.ylabel("parallel efficiency (%)")
    plt.title("Parallel efficiency E(N) = S(N)/N")
    plt.xticks(sorted(args.cores))
    plt.legend()
    plt.grid(alpha=0.4, linestyle="--")
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "plot2_efficiency.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"plot: {p2}")

    # shuffle partitions
    if shuffle_records:
        d = pd.DataFrame(shuffle_records)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].bar(d["shuffle_partitions"].astype(str), d["wall_time_sec"],
                  color="steelblue")
        ax[0].set_xlabel("spark.sql.shuffle.partitions")
        ax[0].set_ylabel("wall-clock time (s)")
        ax[0].set_title("Runtime vs shuffle partitions")
        ax[1].bar(d["shuffle_partitions"].astype(str), d["num_tasks"],
                  color="indianred")
        ax[1].set_xlabel("spark.sql.shuffle.partitions")
        ax[1].set_ylabel("scheduled tasks")
        ax[1].set_title("Tasks scheduled")
        for a in ax:
            a.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        p3 = os.path.join(plots_dir, "plot3_shuffle_partitions.png")
        plt.savefig(p3, dpi=200)
        plt.close()
        print(f"plot: {p3}")

    # time breakdown at max cores
    mc = max(args.cores)
    s6 = summary[summary["cores"] == mc].copy()
    if not s6.empty:
        s6["cpu_per_core"] = s6["cpu_time_mean"] / mc
        s6["io_wait"] = np.maximum(0, s6["wall_time_mean"] - s6["cpu_per_core"]
                                   - s6["gc_time_mean"])
        idx = np.arange(len(s6))
        plt.figure(figsize=(10, 4.5))
        plt.barh(idx, s6["cpu_per_core"], label="CPU (per-core equivalent)",
                 color="#1f77b4")
        plt.barh(idx, s6["gc_time_mean"], left=s6["cpu_per_core"],
                 label="JVM GC", color="#ff7f0e")
        plt.barh(idx, s6["io_wait"], left=s6["cpu_per_core"] + s6["gc_time_mean"],
                 label="I/O, shuffle, wait", color="#2ca02c")
        plt.yticks(idx, s6["task_name"])
        plt.xlabel("seconds")
        plt.title(f"Where the time goes, at {mc} cores")
        plt.legend()
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        p4 = os.path.join(plots_dir, "plot4_time_breakdown.png")
        plt.savefig(p4, dpi=200)
        plt.close()
        print(f"plot: {p4}")

    print("\nDONE")


if __name__ == "__main__":
    main()
