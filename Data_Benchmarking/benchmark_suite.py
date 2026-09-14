#!/usr/bin/env python3
"""
CloudVeneto Spark Distributed Benchmark Suite (Data Benchmarking)
Author: Group 5 - MAPD Part B

Benchmarks the column-addition tasks from project.ipynb on the Spark cluster.
Data reading and BitString alarm conversion are excluded: the parquet dataset
is read once as a pre-benchmark setup step, and then only the following
transformation tasks are measured:

  Task 1 - is_overheated flag    : substring-based alarm detection (Map / Narrow)
  Task 2 - Timestamp conversion  : UNIX ms -> TimestampType + date_trunc bucket (Map / Narrow)
  Task 3 - Device-level flag     : Window max propagation of is_overheated per device/bucket (Window)
  Task 4 - Metric aggregation    : groupBy pivot with mean/max per metric category (Shuffle / Wide)

Metrics extracted via Spark Web UI REST API (Duration, CPU Time, JVM GC Time, Shuffle I/O).
Strong Scaling tested across 1, 2, 4, 6 cores.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
from pyspark.sql import Window


def get_spark_session(app_name="MAPD_DataBenchmark", cores_max=6, shuffle_partitions=12):
    """Initializes a SparkSession with dedicated core and memory allocations."""
    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_KEY") or os.getenv("S3_PRIVATE_KEY")

    builder = (
        SparkSession.builder
        .master("spark://master:7077")
        .appName(f"{app_name}_Cores_{cores_max}")
        .config("spark.cores.max", str(cores_max))
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "1536m")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.1,org.apache.hadoop:hadoop-common:3.4.1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.access.key", access_key or "")
        .config("spark.hadoop.fs.s3a.secret.key", secret_key or "")
        .config("spark.hadoop.fs.s3a.endpoint", "https://cloud-areapd.pd.infn.it:5210")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.metadatastore.impl", "org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("com.amazonaws.sdk.disableCertChecking", "true")
    )
    return builder.getOrCreate()


def get_stage_metrics(app_id, min_stage_id=0):
    """Queries the Spark UI REST API to extract stage-level performance counters."""
    url = f"http://localhost:4040/api/v1/applications/{app_id}/stages"
    try:
        with urllib.request.urlopen(url, timeout=5) as res:
            stages = json.loads(res.read().decode())
    except Exception as e:
        print(f"Warning: could not query Spark REST API: {e}", flush=True)
        return {
            "executorRunTime_sec": 0.0,
            "executorCpuTime_sec": 0.0,
            "jvmGcTime_sec": 0.0,
            "shuffleReadBytes": 0,
            "shuffleWriteBytes": 0,
            "memoryBytesSpilled": 0,
            "diskBytesSpilled": 0,
            "numTasks": 0,
            "max_stage_id": min_stage_id,
        }

    relevant_stages = [s for s in stages if s.get("stageId", -1) >= min_stage_id]
    max_id = max([s.get("stageId", -1) for s in stages], default=min_stage_id)

    run_time = sum(s.get("executorRunTime", 0) for s in relevant_stages) / 1000.0
    cpu_time = sum(s.get("executorCpuTime", 0) for s in relevant_stages) / 1e9
    gc_time = sum(s.get("jvmGcTime", 0) for s in relevant_stages) / 1000.0
    shuff_read = sum(s.get("shuffleReadBytes", 0) for s in relevant_stages)
    shuff_write = sum(s.get("shuffleWriteBytes", 0) for s in relevant_stages)
    mem_spill = sum(s.get("memoryBytesSpilled", 0) for s in relevant_stages)
    disk_spill = sum(s.get("diskBytesSpilled", 0) for s in relevant_stages)
    tasks = sum(s.get("numTasks", 0) for s in relevant_stages)

    return {
        "executorRunTime_sec": run_time,
        "executorCpuTime_sec": cpu_time,
        "jvmGcTime_sec": gc_time,
        "shuffleReadBytes": shuff_read,
        "shuffleWriteBytes": shuff_write,
        "memoryBytesSpilled": mem_spill,
        "diskBytesSpilled": disk_spill,
        "numTasks": tasks,
        "max_stage_id": max_id,
    }


def get_current_max_stage(app_id):
    """Returns the highest stage ID recorded so far."""
    url = f"http://localhost:4040/api/v1/applications/{app_id}/stages"
    try:
        with urllib.request.urlopen(url, timeout=5) as res:
            stages = json.loads(res.read().decode())
            return max([s.get("stageId", -1) for s in stages], default=0) + 1
    except Exception:
        return 0


# ==============================================================================
# BENCHMARK TASK DEFINITIONS  (from project.ipynb — column-addition cells only)
# ==============================================================================

def run_task1_is_overheated(df_spark):
    """
    Task 1 - is_overheated flag  (Map / Narrow Dependency, Zero Shuffle)
    -----------------------------------------------------------------------
    Reproduces the cell in project.ipynb that adds the 'is_overheated' boolean
    column by checking positions 9-11 of the BitString column for A5/A9 metrics.

    Note: df_spark is expected to already have the 'BitString' column (produced
    during the pre-benchmark setup, NOT benchmarked here).

    The .filter(isNotNull) on the newly computed column prevents Catalyst's
    Column Pruning rule from eliminating the withColumn projection entirely:
    a plain .count() on BooleanType (never NULL) would let Catalyst drop the
    column from the physical plan and skip the actual sf.substring computation.
    """
    df_out = df_spark.withColumn(
        "is_overheated",
        sf.when(
            df_spark.metric.isin("A5", "A9") &
            (sf.substring(df_spark.BitString, 9, 3) != "000"),
            True
        ).otherwise(False)
    )
    # isNotNull forces Catalyst to retain the 'is_overheated' projection in the
    # physical plan, ensuring the substring/bin/lpad work is actually executed.
    count = df_out.filter(sf.col("is_overheated").isNotNull()).count()
    return count, df_out


def run_task2_timestamp_and_bucket(df_spark):
    """
    Task 2 - Timestamp conversion + date_trunc bucket  (Map / Narrow)
    ------------------------------------------------------------------
    Reproduces the cell that overwrites 'when' with a proper TimestampType
    via timestamp_millis() and adds 'timestamp_bucket' via date_trunc('minute').

    The .filter(isNotNull) on 'timestamp_bucket' prevents Catalyst's Column
    Pruning rule from discarding the timestamp_millis / date_trunc projection
    when the downstream action is a plain count().
    """
    df_out = df_spark.withColumn(
        "when",
        sf.timestamp_millis(df_spark.when)
    ).withColumn(
        "timestamp_bucket",
        sf.date_trunc("minute", "when")
    )
    # isNotNull on 'timestamp_bucket' ensures the date_trunc (and the upstream
    # timestamp_millis cast) are retained in the Catalyst physical plan.
    count = df_out.filter(sf.col("timestamp_bucket").isNotNull()).count()
    return count, df_out


def run_task3_device_flag(df_spark):
    """
    Task 3 - Device-level is_overheated propagation  (Window / Wide Dependency)
    ---------------------------------------------------------------------------
    Reproduces the Window.partitionBy('timestamp_bucket', 'hwid') cell that
    propagates the is_overheated flag across all rows in the same device/bucket.

    Expects df_spark to have: 'timestamp_bucket', 'hwid', 'is_overheated'.

    Window.partitionBy("timestamp_bucket", "hwid") is a WIDE dependency:
    Spark must perform a Hash Shuffle Exchange to co-locate all rows sharing the
    same (timestamp_bucket, hwid) pair on the same executor before it can compute
    max(is_overheated) locally.  There is NO partition-local shortcut.

    The .filter(isNotNull) on 'is_overheated_device' prevents Catalyst's Column
    Pruning rule from eliminating the entire Window + ShuffleExchange from the
    physical plan when the action is a plain count().
    """
    w_device = Window.partitionBy("timestamp_bucket", "hwid")
    df_out = df_spark.withColumn(
        "is_overheated_device",
        sf.max(sf.col("is_overheated")).over(w_device)
    )
    # isNotNull on 'is_overheated_device' forces Catalyst to retain the Window
    # expression AND the upstream Hash Shuffle Exchange in the physical plan.
    count = df_out.filter(sf.col("is_overheated_device").isNotNull()).count()
    return count, df_out


def run_task4_metric_aggregation(df_spark):
    """
    Task 4 - Metric aggregation via groupBy  (Shuffle / Wide Dependency)
    ---------------------------------------------------------------------
    Reproduces the final groupBy cell in project.ipynb:
      - continuous metrics -> mean
      - all others (discrete, alarms, counters) -> max
    Uses coalesce to pick mean when applicable, else falls back to max.

    Expects df_spark to have: 'timestamp_bucket', 'hwid', 'metric', 'value',
    and 'is_overheated_device' (or 'is_overheated' as fallback).
    """
    continuous_metrics = [
        "E1", "E2",
        "S19", "S37", "S39", "S40", "S41", "S42", "S43", "S45", "S46", "S47", "S49", "S50",
        "S69", "S70", "S71", "S72", "S80", "S81", "S83", "S86", "S90", "S94", "S97",
        "S100", "S101", "S102", "S106", "S107", "S108", "S109", "S110",
        "S122", "S124", "S125", "S126", "S128", "S129",
        "S137", "S138", "S140", "S143", "S147",
        "S151", "S154", "S157", "S158", "S159",
        "S163", "S164", "S165", "S166", "S167",
        "S178", "S180", "S181",
    ]

    # Use is_overheated_device if available (produced by Task 3), else is_overheated
    flag_col = "is_overheated_device" if "is_overheated_device" in df_spark.columns else "is_overheated"

    df_aggregated = (
        df_spark.groupBy("timestamp_bucket", "hwid", "metric")
        .agg(
            sf.coalesce(
                sf.mean(sf.when(sf.col("metric").isin(continuous_metrics), sf.col("value"))),
                sf.max(sf.col("value"))
            ).alias("aggregated_value"),
            sf.first(flag_col).alias("is_overheated")
        )
    )
    count = df_aggregated.count()
    return count, df_aggregated


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CloudVeneto Spark Data Benchmark (project.ipynb tasks)")
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 6],
                        help="List of core allocations to test")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of repetitions per benchmark")
    parser.add_argument("--shuffle-test", action="store_true", default=True,
                        help="Also test shuffle partitions tuning (6, 12, 32, 200)")
    parser.add_argument("--output-dir", type=str,
                        default="/opt/mapd-project/Data_Benchmarking_Final",
                        help="Output directory for CSV, JSON and plots")
    args = parser.parse_args()

    s3_parquet_path = "s3a://MAPDB-Group5/data_parquet"
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "benchmark_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING DATA BENCHMARK SUITE (project.ipynb column-addition tasks)")
    print(f"Data source       : {s3_parquet_path}")
    print(f"Cores configs     : {args.cores}")
    print(f"Repetitions/test  : {args.repeats}")
    print(f"Output directory  : {output_dir}")
    print("=" * 80, flush=True)

    benchmark_records = []

    for n_cores in args.cores:
        print(f"\n>>> Initializing Spark Session - spark.cores.max = {n_cores} ...", flush=True)
        spark = get_spark_session(cores_max=n_cores, shuffle_partitions=12)
        app_id = spark.sparkContext.applicationId
        print(f"    Application ID: {app_id}")

        # ------------------------------------------------------------------
        # PRE-BENCHMARK SETUP  (not measured):
        #   1. Read parquet from S3
        #   2. Add BitString column  (alarm conversion - excluded per spec)
        #   3. Cache + materialize so Task 1 only measures its own transformation
        # ------------------------------------------------------------------
        print("    [Setup] Reading parquet data from S3 ...", flush=True)
        df_raw = spark.read.parquet(s3_parquet_path)

        print("    [Setup] Adding BitString column and caching (not benchmarked) ...", flush=True)
        df_with_bitstring = df_raw.withColumn(
            "BitString",
            sf.when(
                df_raw.metric.isin("A5", "A9"),
                sf.lpad(sf.bin(df_raw.value), 16, "0")
            ).otherwise(df_raw.value.cast("string"))
        ).cache()
        df_with_bitstring.count()  # materialise into memory — S3 read happens here, NOT during benchmarks

        # ------------------------------------------------------------------
        # TASK 1 - is_overheated flag
        # Input: df_with_bitstring (cached) → measures only withColumn + count
        # ------------------------------------------------------------------
        df_t1_result = None
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, df_t1_result = run_task1_is_overheated(df_with_bitstring)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 1: is_overheated Flag (Substring BitString)",
                "task_category": "Map / Narrow",
                "cores": n_cores,
                "repeat": r + 1,
                "wall_time_sec": wall_time,
                "cpu_time_sec": metrics["executorCpuTime_sec"],
                "run_time_sec": metrics["executorRunTime_sec"],
                "gc_time_sec": metrics["jvmGcTime_sec"],
                "shuffle_read_bytes": metrics["shuffleReadBytes"],
                "shuffle_write_bytes": metrics["shuffleWriteBytes"],
                "num_tasks": metrics["numTasks"],
                "result_count": res,
            }
            benchmark_records.append(rec)
            print(f"    [Task1 | Cores: {n_cores} | Rep: {r+1}] "
                  f"Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | "
                  f"GC: {rec['gc_time_sec']:.3f}s | Rows: {res}", flush=True)

        # Cache Task 1 output so Task 2 only measures its own transformation
        print("    [Setup] Caching Task 1 output for Task 2 ...", flush=True)
        df_t1_cached = df_t1_result.cache()
        df_t1_cached.count()  # materialise

        # ------------------------------------------------------------------
        # TASK 2 - Timestamp conversion + date_trunc bucket
        # Input: df_t1_cached → measures only withColumn x2 + count
        # ------------------------------------------------------------------
        df_t2_result = None
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, df_t2_result = run_task2_timestamp_and_bucket(df_t1_cached)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 2: Timestamp + date_trunc Bucket",
                "task_category": "Map / Narrow",
                "cores": n_cores,
                "repeat": r + 1,
                "wall_time_sec": wall_time,
                "cpu_time_sec": metrics["executorCpuTime_sec"],
                "run_time_sec": metrics["executorRunTime_sec"],
                "gc_time_sec": metrics["jvmGcTime_sec"],
                "shuffle_read_bytes": metrics["shuffleReadBytes"],
                "shuffle_write_bytes": metrics["shuffleWriteBytes"],
                "num_tasks": metrics["numTasks"],
                "result_count": res,
            }
            benchmark_records.append(rec)
            print(f"    [Task2 | Cores: {n_cores} | Rep: {r+1}] "
                  f"Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | "
                  f"GC: {rec['gc_time_sec']:.3f}s | Rows: {res}", flush=True)

        # Cache Task 2 output so Task 3 only measures its own transformation
        print("    [Setup] Caching Task 2 output for Task 3 ...", flush=True)
        df_t2_cached = df_t2_result.cache()
        df_t2_cached.count()  # materialise

        # ------------------------------------------------------------------
        # TASK 3 - Device-level flag propagation (Window)
        # Input: df_t2_cached → measures only withColumn(Window) + count
        # ------------------------------------------------------------------
        df_t3_result = None
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, df_t3_result = run_task3_device_flag(df_t2_cached)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 3: Device Flag Propagation (Window)",
                "task_category": "Window / Skew",
                "cores": n_cores,
                "repeat": r + 1,
                "wall_time_sec": wall_time,
                "cpu_time_sec": metrics["executorCpuTime_sec"],
                "run_time_sec": metrics["executorRunTime_sec"],
                "gc_time_sec": metrics["jvmGcTime_sec"],
                "shuffle_read_bytes": metrics["shuffleReadBytes"],
                "shuffle_write_bytes": metrics["shuffleWriteBytes"],
                "num_tasks": metrics["numTasks"],
                "result_count": res,
            }
            benchmark_records.append(rec)
            print(f"    [Task3 | Cores: {n_cores} | Rep: {r+1}] "
                  f"Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | "
                  f"GC: {rec['gc_time_sec']:.3f}s | Rows: {res}", flush=True)

        # Cache Task 3 output so Task 4 only measures its own transformation
        print("    [Setup] Caching Task 3 output for Task 4 ...", flush=True)
        df_t3_cached = df_t3_result.cache()
        df_t3_cached.count()  # materialise
        # Wait for the Spark REST API to flush the cache-materialisation stages
        # before snapshotting the stage counter for Task 4.  Without this pause
        # the cache-count stages may still be 'RUNNING' in the API response and
        # get folded into Task 4's shuffle metrics, producing anomalously low
        # shuffle_read_bytes (e.g. 708 B instead of ~807 MB).
        time.sleep(1)

        # ------------------------------------------------------------------
        # TASK 4 - Metric aggregation (groupBy - Shuffle / Wide)
        # Input: df_t3_cached → measures only groupBy + agg + count
        # ------------------------------------------------------------------
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, _ = run_task4_metric_aggregation(df_t3_cached)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 4: Metric Aggregation (groupBy mean/max)",
                "task_category": "Shuffle / Wide",
                "cores": n_cores,
                "repeat": r + 1,
                "wall_time_sec": wall_time,
                "cpu_time_sec": metrics["executorCpuTime_sec"],
                "run_time_sec": metrics["executorRunTime_sec"],
                "gc_time_sec": metrics["jvmGcTime_sec"],
                "shuffle_read_bytes": metrics["shuffleReadBytes"],
                "shuffle_write_bytes": metrics["shuffleWriteBytes"],
                "num_tasks": metrics["numTasks"],
                "result_count": res,
            }
            benchmark_records.append(rec)
            print(f"    [Task4 | Cores: {n_cores} | Rep: {r+1}] "
                  f"Wall: {wall_time:.3f}s | ShuffRead: {rec['shuffle_read_bytes'] / (1024**2):.1f}MB | "
                  f"Rows: {res}", flush=True)

        # Free cached DataFrames before stopping Spark
        df_with_bitstring.unpersist()
        df_t1_cached.unpersist()
        df_t2_cached.unpersist()
        df_t3_cached.unpersist()

        spark.stop()
        time.sleep(2)

    # --------------------------------------------------------------------------
    # OPTIONAL: SHUFFLE PARTITIONS TUNING AT MAX CORES = 6
    # (runs Task 4 - the heaviest shuffle-bound task - at varying partition counts)
    # --------------------------------------------------------------------------
    shuffle_records = []
    if args.shuffle_test and 6 in args.cores:
        print("\n" + "=" * 80)
        print("RUNNING SPARK SQL SHUFFLE PARTITIONS TUNING (CORES = 6, Task 4)")
        print("=" * 80, flush=True)
        partition_options = [6, 12, 32, 200]

        for p in partition_options:
            print(f"\n>>> Testing spark.sql.shuffle.partitions = {p} ...", flush=True)
            spark = get_spark_session(cores_max=6, shuffle_partitions=p)
            app_id = spark.sparkContext.applicationId

            # Rebuild and cache the pipeline up to Task 3 (setup, not measured)
            df_raw = spark.read.parquet(s3_parquet_path)
            df_bs = df_raw.withColumn(
                "BitString",
                sf.when(df_raw.metric.isin("A5", "A9"),
                        sf.lpad(sf.bin(df_raw.value), 16, "0")
                        ).otherwise(df_raw.value.cast("string"))
            ).cache()
            df_bs.count()  # materialise S3 read
            _, df_t1 = run_task1_is_overheated(df_bs)
            df_t1_c = df_t1.cache(); df_t1_c.count()
            _, df_t2 = run_task2_timestamp_and_bucket(df_t1_c)
            df_t2_c = df_t2.cache(); df_t2_c.count()
            _, df_t3 = run_task3_device_flag(df_t2_c)
            df_t3_c = df_t3.cache(); df_t3_c.count()
            time.sleep(1)  # allow REST API to flush cache stages before Task 4 snapshot

            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, _ = run_task4_metric_aggregation(df_t3_c)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "shuffle_partitions": p,
                "wall_time_sec": wall_time,
                "cpu_time_sec": metrics["executorCpuTime_sec"],
                "gc_time_sec": metrics["jvmGcTime_sec"],
                "shuffle_read_bytes": metrics["shuffleReadBytes"],
                "shuffle_write_bytes": metrics["shuffleWriteBytes"],
                "num_tasks": metrics["numTasks"],
            }
            shuffle_records.append(rec)
            print(f"    [ShufflePartitions: {p}] "
                  f"Wall: {wall_time:.3f}s | Tasks: {rec['num_tasks']} | "
                  f"ShuffRead: {rec['shuffle_read_bytes'] / (1024**2):.1f}MB", flush=True)
            df_bs.unpersist()
            df_t1_c.unpersist()
            df_t2_c.unpersist()
            df_t3_c.unpersist()
            spark.stop()
            time.sleep(2)

    # ==========================================================================
    # DATA AGGREGATION, SPEEDUP & EFFICIENCY COMPUTATION
    # ==========================================================================
    df_results = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    json_path = os.path.join(output_dir, "benchmark_results.json")

    # Group by task and cores to get mean and std
    summary = df_results.groupby(["task_name", "task_category", "cores"]).agg({
        "wall_time_sec": ["mean", "std"],
        "cpu_time_sec": ["mean"],
        "gc_time_sec": ["mean"],
        "shuffle_read_bytes": ["mean"],
        "shuffle_write_bytes": ["mean"],
    }).reset_index()

    summary.columns = [
        "task_name", "task_category", "cores",
        "wall_time_mean", "wall_time_std", "cpu_time_mean",
        "gc_time_mean", "shuffle_read_bytes_mean", "shuffle_write_bytes_mean"
    ]

    # Compute Speedup and Efficiency relative to 1 Core
    base_times = summary[summary["cores"] == 1].set_index("task_name")["wall_time_mean"].to_dict()

    summary["speedup"] = summary.apply(
        lambda row: base_times.get(row["task_name"], row["wall_time_mean"]) / row["wall_time_mean"] if row["wall_time_mean"] > 0 else 1.0,
        axis=1
    )
    summary["efficiency"] = summary["speedup"] / summary["cores"]

    # Save summary tables
    summary.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({
            "raw_records": benchmark_records,
            "summary": summary.to_dict(orient="records"),
            "shuffle_tuning": shuffle_records
        }, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (STRONG SCALING & METRICS)")
    print("=" * 80)
    print(summary[["task_name", "cores", "wall_time_mean", "speedup", "efficiency", "cpu_time_mean", "gc_time_mean"]].to_string())
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")

    # ==========================================================================
    # GENERATE PUBLICATION-QUALITY PLOTS
    # ==========================================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --------------------------------------------------------------------------
    # Plot 1: Wall-Clock Time per Task across Core Counts
    # --------------------------------------------------------------------------
    plt.figure(figsize=(13, 6))
    tasks_unique = summary["task_name"].unique()
    palette = sns.color_palette("tab10", len(tasks_unique))

    for i, tname in enumerate(tasks_unique):
        sub = summary[summary["task_name"] == tname].sort_values("cores")
        plt.plot(sub["cores"], sub["wall_time_mean"], marker="o", linewidth=2.2,
                 label=tname, color=palette[i])
        plt.fill_between(sub["cores"],
                         sub["wall_time_mean"] - sub["wall_time_std"].fillna(0),
                         sub["wall_time_mean"] + sub["wall_time_std"].fillna(0),
                         alpha=0.15, color=palette[i])

    plt.title("Wall-Clock Time per Task (Strong Scaling)", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Cores (N)", fontsize=12)
    plt.ylabel("Wall-Clock Time (s)", fontsize=12)
    plt.xticks(args.cores)
    plt.yscale("log")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6, which="both")
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "plot1_wall_time_per_task.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 1: {plot1_path}")

    # --------------------------------------------------------------------------
    # Plot 2: Strong Scaling Speedup across Tasks
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    ideal_cores = np.array(args.cores)
    plt.plot(ideal_cores, ideal_cores, "k--", label="Ideal Linear Speedup", linewidth=1.5, alpha=0.7)

    tasks_unique = summary["task_name"].unique()
    palette = sns.color_palette("tab10", len(tasks_unique))

    for i, tname in enumerate(tasks_unique):
        sub = summary[summary["task_name"] == tname].sort_values("cores")
        plt.plot(sub["cores"], sub["speedup"], marker="o", linewidth=2.2, label=tname, color=palette[i])

    plt.title("Strong Scaling Speedup S(N) across Spark Computational Tasks", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Cores (N)", fontsize=12)
    plt.ylabel("Speedup S(N) = T(1) / T(N)", fontsize=12)
    plt.xticks(args.cores)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plot2_path = os.path.join(plots_dir, "plot2_strong_scaling_speedup.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 2: {plot2_path}")

    # --------------------------------------------------------------------------
    # Plot 3: Parallel Efficiency
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.axhline(100.0, color="k", linestyle="--", label="Ideal 100% Efficiency", alpha=0.7)

    for i, tname in enumerate(tasks_unique):
        sub = summary[summary["task_name"] == tname].sort_values("cores")
        plt.plot(sub["cores"], sub["efficiency"] * 100.0, marker="s", linewidth=2.2, label=tname, color=palette[i])

    plt.title("Parallel Efficiency E(N) = S(N)/N across Spark Tasks", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Cores (N)", fontsize=12)
    plt.ylabel("Parallel Efficiency (%)", fontsize=12)
    plt.xticks(args.cores)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plot3_path = os.path.join(plots_dir, "plot3_parallel_efficiency.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Plot 3: {plot3_path}")

    # --------------------------------------------------------------------------
    # Plot 4: Spark SQL Shuffle Partitions Tuning
    # --------------------------------------------------------------------------
    if shuffle_records:
        df_shuff = pd.DataFrame(shuffle_records)
        plt.figure(figsize=(11, 5))

        plt.subplot(1, 2, 1)
        sns.barplot(data=df_shuff, x="shuffle_partitions", y="wall_time_sec", palette="viridis")
        plt.title("Execution Time vs spark.sql.shuffle.partitions", fontsize=12, fontweight="bold")
        plt.xlabel("Shuffle Partitions Count")
        plt.ylabel("Wall-Clock Time (s)")

        plt.subplot(1, 2, 2)
        sns.barplot(data=df_shuff, x="shuffle_partitions", y="num_tasks", palette="rocket")
        plt.title("Total Scheduled Spark Tasks", fontsize=12, fontweight="bold")
        plt.xlabel("Shuffle Partitions Count")
        plt.ylabel("Number of Tasks")

        plt.tight_layout()
        plot4_path = os.path.join(plots_dir, "plot4_shuffle_partitions_tuning.png")
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        print(f"Generated Plot 4: {plot4_path}")

    # --------------------------------------------------------------------------
    # Plot 5: Execution Breakdown (CPU vs GC vs I/O/Wait) at 6 Cores
    # --------------------------------------------------------------------------
    summary_6c = summary[summary["cores"] == 6].copy()
    if not summary_6c.empty:
        summary_6c["io_wait_time"] = np.maximum(0, summary_6c["wall_time_mean"] - (summary_6c["cpu_time_mean"] / 6.0) - summary_6c["gc_time_mean"])
        
        plt.figure(figsize=(12, 6))
        bar_data = summary_6c.set_index("task_name")[["cpu_time_mean", "gc_time_mean", "io_wait_time"]]
        bar_data["norm_cpu"] = summary_6c.set_index("task_name")["cpu_time_mean"] / 6.0
        
        plot_df = pd.DataFrame({
            "Active CPU (Per-Core Equiv)": bar_data["norm_cpu"],
            "JVM GC Overhead": bar_data["gc_time_mean"],
            "Network / I/O / Shuffle Wait": bar_data["io_wait_time"]
        })
        
        plot_df.plot(kind="barh", stacked=True, color=["#1f77b4", "#ff7f0e", "#2ca02c"], figsize=(12, 6))
        plt.title("Execution Time Breakdown at 6 Cores (CloudVeneto Cluster)", fontsize=13, fontweight="bold")
        plt.xlabel("Effective Duration (Seconds)")
        plt.ylabel("")
        plt.legend(loc="lower right")
        plt.tight_layout()

        plot5_path = os.path.join(plots_dir, "plot5_metrics_breakdown.png")
        plt.savefig(plot5_path, dpi=300)
        plt.close()
        print(f"Generated Plot 5: {plot5_path}")

    print("\nALL BENCHMARKS AND VISUALIZATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
