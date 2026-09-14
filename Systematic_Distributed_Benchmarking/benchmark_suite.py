#!/usr/bin/env python3
"""
CloudVeneto Spark Distributed Benchmark Suite (Task 3.3.3 - Predictive Maintenance)
Author: Nicola Lavarda (Group 5 - MAPD Part B)

Measures distributed execution metrics according to Apache Spark Monitoring & Metrics:
- Macro-computational tasks (Map-bound, Shuffle-bound, Window/Skew-bound, Iterative ML)
- Strong Scaling (Speedup & Parallel Efficiency across 1, 2, 4, 6 cores)
- Bitwise Masking vs String Conversion comparison
- Spark SQL Shuffle Partitions Tuning (6, 12, 32, 200)
- Execution metrics extracted via Spark Web UI REST API (Duration, CPU Time, JVM GC Time, Shuffle I/O)
"""

import os
import sys
import time
import json
import argparse
import logging
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF

logger = logging.getLogger("SparkBenchmark")


def setup_logger(output_dir, log_file=None):
    """Configures a logger that outputs to both stdout and a persistent file."""
    os.makedirs(output_dir, exist_ok=True)
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"benchmark_{timestamp}.log")
    elif not os.path.isabs(log_file):
        log_file = os.path.join(output_dir, log_file)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def get_spark_session(app_name="MAPD_Benchmark", cores_max=6, shuffle_partitions=12):
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
        logger.warning(f"Could not query Spark REST API: {e}")
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
# BENCHMARK TASK DEFINITIONS
# ==============================================================================

def run_task1_map_bound(spark, s3_parquet_path, mode="bitwise"):
    """
    Task 1: Map-Bound (Narrow Dependency, Zero Shuffle).
    Compares Bitwise Masking (Mask 224) vs String Conversion (lpad + substring).
    """
    df = spark.read.parquet(s3_parquet_path)

    if mode == "bitwise":
        df_out = df.withColumn(
            "is_overheated",
            sf.when(
                sf.col("metric").isin("A5", "A9") & (sf.col("value").cast("long").bitwiseAND(224) > 0),
                sf.lit(1)
            ).otherwise(sf.lit(0))
        )
    elif mode == "string":
        df_out = df.withColumn(
            "BitString",
            sf.when(
                sf.col("metric").isin("A5", "A9"),
                sf.lpad(sf.bin(sf.col("value").cast("long")), 16, "0")
            ).otherwise(sf.lit(None))
        ).withColumn(
            "is_overheated",
            sf.when(
                sf.col("metric").isin("A5", "A9") & (sf.substring(sf.col("BitString"), 9, 3) != "000"),
                sf.lit(1)
            ).otherwise(sf.lit(0))
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    count = df_out.filter(sf.col("is_overheated") == 1).count()
    return count


def run_task2_shuffle_bound(spark, s3_parquet_path):
    """
    Task 2: Shuffle-Bound (Wide Dependency, Network & Disk Exchange).
    1-Minute Temporal Resampling & Memory-Safe Wide Pivoting for 30 domain metrics.
    """
    continuous_features = [
        "S41", "S100", "S101", "S102", "S106", "S107", "S108", "S109", "S110",
        "S124", "S125", "S126", "S157", "S158", "S159", "S163", "S164", "S165",
        "S166", "S167", "S180", "S181", "S81", "S138", "S39", "S40"
    ]
    discrete_features = ["S117", "S118", "S169", "S170"]
    all_metrics = continuous_features + discrete_features

    df = spark.read.parquet(s3_parquet_path)
    df_decoded = df.withColumn(
        "is_overheated",
        sf.when(
            sf.col("metric").isin("A5", "A9") & (sf.col("value").cast("long").bitwiseAND(224) > 0),
            sf.lit(1)
        ).otherwise(sf.lit(0))
    ).withColumn(
        "timestamp_bucket",
        sf.date_trunc("minute", sf.timestamp_millis(sf.col("when").cast("long")))
    )

    w_device = Window.partitionBy("hwid", "timestamp_bucket")
    df_flagged = df_decoded.withColumn("device_overheated", sf.max("is_overheated").over(w_device))
    df_sub = df_flagged.filter(sf.col("metric").isin(all_metrics))

    pivot_expressions = [
        sf.mean(sf.when(sf.col("metric") == m, sf.col("value").cast("double"))).alias(m)
        for m in continuous_features
    ] + [
        sf.max(sf.when(sf.col("metric") == m, sf.col("value").cast("double"))).alias(m)
        for m in discrete_features
    ] + [
        sf.max("device_overheated").alias("is_overheated")
    ]

    df_pivoted = df_sub.groupBy("hwid", "timestamp_bucket").agg(*pivot_expressions).cache()
    count = df_pivoted.count()
    return count, df_pivoted


def run_task3_skew_window_bound(spark, df_pivoted):
    """
    Task 3: Skew & Window-Bound (Stateful Cumulative Window).
    Forward-Fill missing values across device partitions (Window.partitionBy('hwid')).
    Demonstrates parallelism limitation since cardinality(hwid) == 4 (max 4 concurrent tasks).
    """
    w_ffill = (
        Window.partitionBy("hwid")
        .orderBy("timestamp_bucket")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df_filled = df_pivoted.withColumn(
        "S109_filled",
        sf.last("S109", ignorenulls=True).over(w_ffill)
    ).withColumn(
        "S41_filled",
        sf.last("S41", ignorenulls=True).over(w_ffill)
    ).cache()

    count = df_filled.filter(sf.col("S41_filled").isNotNull()).count()
    return count, df_filled


def run_task4_distributed_mllib(spark, df_filled):
    """
    Task 4: Distributed MLlib Training (Iterative Multi-Pass Histogram Aggregation).
    Trains a distributed RandomForestClassifier (15 trees, depth 5) across worker nodes.
    """
    feature_cols = ["S41_filled", "S109_filled"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df_ml = assembler.transform(df_filled).select(
        "features",
        sf.col("is_overheated").cast("double").alias("label")
    )

    # Balanced sample cached to memory to benchmark distributed tree split search without lineage recomputation
    df_train = df_ml.sampleBy("label", fractions={0.0: 0.10, 1.0: 1.0}, seed=42).cache()
    df_train.count()

    rf = SparkRF(featuresCol="features", labelCol="label", numTrees=15, maxDepth=5, seed=42)
    model = rf.fit(df_train)
    num_trees = int(model.getNumTrees)
    df_train.unpersist()
    return num_trees


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CloudVeneto Spark Distributed Benchmark")
    parser.add_argument("--cores", nargs="+", type=int, default=[1, 2, 4, 6], help="List of core allocations to test")
    parser.add_argument("--repeats", type=int, default=2, help="Number of repetitions per benchmark")
    parser.add_argument("--shuffle-test", action="store_true", default=True, help="Test shuffle partitions (6, 12, 32, 200)")
    parser.add_argument("--output-dir", type=str, default="/home/nlavarda/AnomalyDection-Spark/Systematic_Distributed_Benchmarking", help="Output directory")
    parser.add_argument("--log-file", type=str, default=None, help="Custom log file name or path (default: auto-generated timestamped log)")
    args = parser.parse_args()

    s3_parquet_path = "s3a://MAPDB-Group5/data_parquet"
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, "benchmark_plots")
    os.makedirs(plots_dir, exist_ok=True)

    log, log_file = setup_logger(output_dir, args.log_file)

    logger.info("=" * 80)
    logger.info("STARTING SYSTEMATIC DISTRIBUTED SPARK BENCHMARK (TASK 3.3.3)")
    logger.info(f"Data source: {s3_parquet_path}")
    logger.info(f"Cores configurations: {args.cores}")
    logger.info(f"Repetitions per test: {args.repeats}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    benchmark_records = []

    # --------------------------------------------------------------------------
    # GLOBAL CLUSTER WARM-UP (S3 connection pool, JVM classes & OS Page Cache)
    # --------------------------------------------------------------------------
    logger.info("\n>>> Performing Global Cluster Warm-Up (6 Cores) to pre-warm OS cache & S3 connections ...")
    spark_warmup = get_spark_session(cores_max=max(args.cores), shuffle_partitions=12)
    try:
        # Scan parquet columns to warm S3 connection pool, HTTPS SSL handshakes and OS page cache
        spark_warmup.read.parquet(s3_parquet_path).select("metric", "value").filter(sf.col("value") == "warmup").count()
        logger.info("    Cluster Warm-Up complete.")
    except Exception as e:
        logger.warning(f"    Warm-up notice: {e}")
    finally:
        spark_warmup.stop()
        time.sleep(2)

    for n_cores in args.cores:
        logger.info(f"\n>>> Initializing Spark Session with spark.cores.max = {n_cores} ...")
        spark = get_spark_session(cores_max=n_cores, shuffle_partitions=12)
        app_id = spark.sparkContext.applicationId
        logger.info(f"    Application ID: {app_id}")

        # Warm-up run to initialize JVM classes, S3A connectors & connection pool
        logger.info("    Executing Warm-up run...")
        try:
            spark.read.parquet(s3_parquet_path).limit(100).count()
        except Exception as e:
            logger.warning(f"    Warm-up error: {e}")

        # ----------------------------------------------------------------------
        # TEST 1A: MAP-BOUND (BITWISE MASK 224)
        # ----------------------------------------------------------------------
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res = run_task1_map_bound(spark, s3_parquet_path, mode="bitwise")
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 1: Map-Bound (Bitwise 224)",
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
            logger.info(f"    [Bitwise | Cores: {n_cores} | Rep: {r+1}] Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | GC: {rec['gc_time_sec']:.3f}s | Result: {res}")

        # ----------------------------------------------------------------------
        # TEST 1B: MAP-BOUND (STRING CONVERSION & SUBSTRING)
        # ----------------------------------------------------------------------
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res = run_task1_map_bound(spark, s3_parquet_path, mode="string")
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 1: Map-Bound (String Substring)",
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
            logger.info(f"    [String  | Cores: {n_cores} | Rep: {r+1}] Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | GC: {rec['gc_time_sec']:.3f}s | Result: {res}")

        # ----------------------------------------------------------------------
        # TEST 2: SHUFFLE-BOUND (RESAMPLING & WIDE PIVOTING)
        # ----------------------------------------------------------------------
        df_pivoted_saved = None
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, df_piv = run_task2_shuffle_bound(spark, s3_parquet_path)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            df_pivoted_saved = df_piv
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 2: Shuffle-Bound (Resample & Pivot)",
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
            logger.info(f"    [Pivot   | Cores: {n_cores} | Rep: {r+1}] Wall: {wall_time:.3f}s | ShuffRead: {rec['shuffle_read_bytes']/(1024**2):.1f}MB | Result: {res}")

            if r < args.repeats - 1:
                df_piv.unpersist()
            else:
                df_pivoted_saved = df_piv

        # ----------------------------------------------------------------------
        # TEST 3: SKEW / WINDOW-BOUND (FORWARD-FILL IMPUTATION)
        # ----------------------------------------------------------------------
        df_filled_saved = None
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, df_fill = run_task3_skew_window_bound(spark, df_pivoted_saved)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            df_filled_saved = df_fill
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 3: Skew/Window (Forward-Fill)",
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
            logger.info(f"    [FFill   | Cores: {n_cores} | Rep: {r+1}] Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | Result: {res}")

            if r < args.repeats - 1:
                df_fill.unpersist()
            else:
                df_filled_saved = df_fill

        # ----------------------------------------------------------------------
        # TEST 4: DISTRIBUTED MLLIB TRAINING (RANDOM FOREST)
        # ----------------------------------------------------------------------
        for r in range(args.repeats):
            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res = run_task4_distributed_mllib(spark, df_filled_saved)
            t1 = time.perf_counter()
            wall_time = t1 - t0
            metrics = get_stage_metrics(app_id, min_stage_id=start_stage)

            rec = {
                "task_name": "Task 4: Distributed MLlib (Random Forest)",
                "task_category": "Distributed ML",
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
            logger.info(f"    [MLlib   | Cores: {n_cores} | Rep: {r+1}] Wall: {wall_time:.3f}s | CPU: {rec['cpu_time_sec']:.3f}s | Trees: {res}")

        spark.stop()
        time.sleep(2)

    # --------------------------------------------------------------------------
    # OPTIONAL: SHUFFLE PARTITIONS TUNING (AT MAX CORES = 6)
    # --------------------------------------------------------------------------
    shuffle_records = []
    if args.shuffle_test and 6 in args.cores:
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING SPARK SQL SHUFFLE PARTITIONS TUNING (CORES = 6)")
        logger.info("=" * 80)
        partition_options = [6, 12, 32, 200]

        for p in partition_options:
            logger.info(f"\n>>> Testing spark.sql.shuffle.partitions = {p} ...")
            spark = get_spark_session(cores_max=6, shuffle_partitions=p)
            app_id = spark.sparkContext.applicationId

            start_stage = get_current_max_stage(app_id)
            t0 = time.perf_counter()
            res, _ = run_task2_shuffle_bound(spark, s3_parquet_path)
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
            logger.info(f"    [ShufflePartitions: {p}] Wall: {wall_time:.3f}s | Tasks: {rec['num_tasks']} | ShuffRead: {rec['shuffle_read_bytes']/(1024**2):.1f}MB")
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

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY (STRONG SCALING & METRICS)")
    logger.info("=" * 80)
    logger.info("\n" + summary[["task_name", "cores", "wall_time_mean", "speedup", "efficiency", "cpu_time_mean", "gc_time_mean"]].to_string())
    logger.info(f"\nSaved CSV to: {csv_path}")
    logger.info(f"Saved JSON to: {json_path}")
    logger.info(f"Saved Log to: {log_file}")

    # ==========================================================================
    # GENERATE PUBLICATION-QUALITY PLOTS
    # ==========================================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --------------------------------------------------------------------------
    # Plot 1: Bitwise Masking vs String Conversion Comparison
    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    df_task1 = summary[summary["task_name"].str.contains("Task 1")]
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_task1, x="cores", y="wall_time_mean", hue="task_name", palette=["#2b5c8f", "#d95f02"])
    plt.title("Execution Time: Bitwise vs String Parsing", fontsize=13, fontweight="bold")
    plt.xlabel("Allocated Cores (CloudVeneto Cluster)")
    plt.ylabel("Wall-Clock Time (s)")
    plt.legend(title="Method", loc="upper right")

    plt.subplot(1, 2, 2)
    sns.barplot(data=df_task1, x="cores", y="gc_time_mean", hue="task_name", palette=["#2b5c8f", "#d95f02"])
    plt.title("JVM Garbage Collection Time Overhead", fontsize=13, fontweight="bold")
    plt.xlabel("Allocated Cores (CloudVeneto Cluster)")
    plt.ylabel("Total JVM GC Time (s)")
    plt.legend(title="Method", loc="upper right")

    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "plot1_bitwise_vs_string.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    logger.info(f"Generated Plot 1: {plot1_path}")

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
    logger.info(f"Generated Plot 2: {plot2_path}")

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
    logger.info(f"Generated Plot 3: {plot3_path}")

    # --------------------------------------------------------------------------
    # Plot 4: Spark SQL Shuffle Partitions Tuning
    # --------------------------------------------------------------------------
    if shuffle_records:
        df_shuff = pd.DataFrame(shuffle_records)
        plt.figure(figsize=(11, 5))

        plt.subplot(1, 2, 1)
        sns.barplot(data=df_shuff, x="shuffle_partitions", y="wall_time_sec", hue="shuffle_partitions", palette="viridis", legend=False)
        plt.title("Execution Time vs spark.sql.shuffle.partitions", fontsize=12, fontweight="bold")
        plt.xlabel("Shuffle Partitions Count")
        plt.ylabel("Wall-Clock Time (s)")

        plt.subplot(1, 2, 2)
        sns.barplot(data=df_shuff, x="shuffle_partitions", y="num_tasks", hue="shuffle_partitions", palette="rocket", legend=False)
        plt.title("Total Scheduled Spark Tasks", fontsize=12, fontweight="bold")
        plt.xlabel("Shuffle Partitions Count")
        plt.ylabel("Number of Tasks")

        plt.tight_layout()
        plot4_path = os.path.join(plots_dir, "plot4_shuffle_partitions_tuning.png")
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        logger.info(f"Generated Plot 4: {plot4_path}")

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
        logger.info(f"Generated Plot 5: {plot5_path}")

    logger.info("\nALL BENCHMARKS AND VISUALIZATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
