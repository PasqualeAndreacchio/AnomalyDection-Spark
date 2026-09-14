### 1. Cosa è successo nel branch di Matteo (La storia dei suoi test)

Matteo ha attraversato due fasi distinte nel tentativo di adattare la tua suite:

1. **Fase 1 (Commit `08f7e96` e `806a389` - Task 5A, 5B, 5C separati):**
   - Matteo ha spezzato il suo task di Anomaly Detection 1 in tre parti (Switch Counts, Group & Join, Correlation).
   - All'interno del ciclo dei repeat ha inserito `.cache()` e `.unpersist()` in un ordine errato: al **Repeat 1** calcolava da zero (es. ~600s), mentre al **Repeat 2** Spark riconosceva lo stesso DAG logico già in memoria e rispondeva istantaneamente in **0.43s** (un finto cache hit).
   - La media tra 600s e 0.4s ha abbassato artificialmente il baseline a 1 core e generato deviazioni standard enormi ($\sigma \approx 296\text{s}$), producendo efficienze fittizie (112%, 133%, 145%).
   - Se ne è accorto e ha scritto `analysis_results.md`.

2. **Fase 2 (Commit `aa23137` e `0507005` - Pipeline Unificata):**
   - Ha eliminato i `.cache()` intermedi creando una pipeline unica (`Correlation Pipeline`), lasciando solo un `.cache()` iniziale su `df_aggregated` fuori dal ciclo.
   - Non ci sono più i cache hit da 0.4s (i 4 repeat sono tutti stabili attorno a 24-25s su 1 core e 9.5-10.5s su 2 core).
   - **Eppure il risultato continua ad essere superlineare:**
     - **1 Core:** $T_1 = 25.65\text{s}$
     - **2 Core:** $T_2 = 10.21\text{s} \longrightarrow \mathbf{Speedup = 2.51\times \ (Efficienza = 125.6\%)}$
     - **4 Core:** $T_4 = 6.74\text{s} \longrightarrow \text{Speedup = } 3.81\times \ (\text{Eff} = 95.2\%)$
     - **6 Core:** $T_6 = 5.03\text{s} \longrightarrow \text{Speedup = } 5.09\times \ (\text{Eff} = 84.9\%)$

---

### 2. Perché a 2 Core ottiene uno Speedup Superlineare (2.51x)?

Non si tratta di un banale errore di formula matematica, ma di **4 precise cause architetturali e metodologiche**:

#### A. Raddoppio della RAM aggregata del cluster (La causa principale)
Nel cluster CloudVeneto ci sono **3 worker nodes da 2 core ciascuno**.
In `get_spark_session`:
```python
builder.config("spark.cores.max", str(cores_max)).config("spark.executor.memory", "2g")
```
Senza specificare `spark.executor.cores`, Spark Standalone con `spark.deploy.spreadOut=true` distribuisce i core su worker diversi:
- Con **`cores_max = 1`**: Spark lancia **1 solo executor** su 1 worker $\rightarrow$ **RAM totale del cluster = 2 GB** (storage memory per il caching $\approx 600\text{MB}$).
- Con **`cores_max = 2`**: Spark alloca 1 core sul Worker 1 e 1 core sul Worker 2 $\rightarrow$ **2 executor** da 2 GB $\rightarrow$ **RAM totale del cluster = 4 GB**!

Matteo ha eseguito `df_aggregated.cache()` prima del benchmark. Con 1 core (2 GB totali), `df_aggregated` soffre di contesa di memoria con l'execution memory di shuffle e join, subisce memory thrashing e manda in crisi il Garbage Collector della JVM (il GC time su 1 core è $0.62\text{s}$, ovvero oltre 10 volte superiore rispetto a 2 core che è $0.05\text{s}$).  
Su 2 core, la memoria aggregata è **raddoppiata a 4 GB**: i dati risiedono completamente in memoria senza memory pressure. Lo speedup è $> 2$ perché **non ha scalato solo le CPU, ha raddoppiato la RAM del cluster**, eliminando il collo di bottiglia di memoria che azzoppava il test a 1 core.

#### B. La Python UDF (`sf.udf`) e il collo di bottiglia del GIL / IPC
Per calcolare i cambi di stato, Matteo ha usato una UDF Python pura:
```python
def count_switches(time_series): ...
count_switches_udf = sf.udf(count_switches, IntegerType())
```
In PySpark, una UDF Python standard non gira nella JVM: i dati vengono serializzati (Py4J/pickle) e inviati via socket a un processo worker Python (`pyspark.daemon`):
- Su **1 core**: c'è **un solo processo Python sequenziale** bloccato dal GIL che gestisce tutta la serializzazione e la logica.
- Su **2 core**: si attivano **due processi Python paralleli** su due nodi fisici distinti, eliminando la serializzazione a collo di bottiglia su singolo processo.

#### C. Carico di lavoro troppo piccolo e Driver Overhead
Guardando i byte scambiati in `benchmark_results.json`:
- La pipeline di Matteo genera solo **2.8 Megabyte di shuffle** e dura appena **5–10 secondi** a 2, 4 e 6 core.
- Su una durata di 5 secondi con 91-107 task, il tempo speso per la coordinazione del Driver, l'handshake REST e lo spawn dei processi Python pesa per 1.5–2 secondi. Questo distorce le curve di scaling classico.

#### D. Perché crolla a 6 Core?
Nella correlazione Matteo raggruppa per device:
```python
df_correlation = df_joined.groupBy("hwid").agg(*corr_exprs)
```
La cardinalità di `hwid` è **pari a 4**. Esattamente come avevi documentato tu per il Task 3, a 6 core Spark può assegnare al massimo 4 partizioni non vuote: **2 core rimangono forzatamente disoccupati**.

---

### 3. Perché nel tuo branch (`nicola-predictive-maintenance`) tutto tornava?

Nel tuo benchmark, ogni singola efficienza è compresa tra il 30% e il 92% (sempre strettamente sublineare $\le 100\%$):
- **Task 1 (String)**: 18.5s $\to$ 10.0s $\to$ 7.2s $\to$ 3.9s (Efficienza: 92% a 2 core, 64% a 4 core, 78% a 6 core).
- **Task 2 (Shuffle Resampling & Wide Pivot)**: 159s $\to$ 99s $\to$ 67s $\to$ 52s (Efficienza: 80% a 2 core, 59% a 4 core, 50% a 6 core).
- **Task 3 (Window skew)**: 4.1s $\to$ 3.6s $\to$ 3.1s $\to$ 2.8s (Efficienza: 57% $\to$ 24%).

**La differenza metodologica fondamentale:**
1. **Carico reale e pesante**: il tuo Task 2 muove **1.33 GB di shuffle** reale e dura fino a 160 secondi. Su un carico di centinaia di secondi, l'overhead di startup e di socket è trascurabile rispetto al calcolo I/O e distribuito.
2. **Nessun pre-caching asimmetrico**: non hai pre-cachato dataset enormi a monte con RAM variabile tra i core.
3. **Nessuna Python UDF**: hai usato operatori nativi di Spark SQL (`sf.bitwiseAND`, `Window`, `VectorAssembler`), compilati direttamente da Catalyst e Tungsten in bytecode Java senza processi Python intermedi.

---

### 4. Cosa consigliare a Matteo per sistemare il suo lavoro

Ecco cosa suggerire a Matteo per uscire da questa situazione:

#### 1. Sostituire la Python UDF con la funzione nativa Spark SQL Window `sf.lag()`
La UDF Python `count_switches` che itera sull'array può essere riscritta interamente con funzioni native:
```python
w = Window.partitionBy("hwid", "metric").orderBy("timestamp_bucket")
df_switches = (
    df_aggregated.filter(sf.col("metric").isin(target_metrics))
    .withColumn("prev_val", sf.lag("aggregated_value", 1).over(w))
    .withColumn(
        "is_switch",
        sf.when(
            sf.col("prev_val").isNotNull() & (sf.col("aggregated_value") != sf.col("prev_val")),
            1,
        ).otherwise(0),
    )
    .withColumn("hour_bucket", sf.date_trunc("hour", sf.col("timestamp_bucket")))
    .groupBy("hour_bucket", "hwid", "metric")
    .agg(sf.sum("is_switch").alias("total_state_switches"))
)
```
*Vantaggi:* Zero processi Python esterni, zero overhead di socket/pickle, ottimizzazione Tungsten nativa in C++/Java bytecode.

#### 2. Non pre-cachare `df_aggregated` fuori dal benchmark
Se vuole misurare lo strong scaling della sua pipeline di Anomaly Detection, deve includere la lettura e l'aggregazione dal parquet grezzo all'interno della misurazione (come hai fatto tu nel Task 2). In questo modo il job richiederà un quantitativo significativo di dati (~1-2 minuti di esecuzione) e non risentirà della discrepanza di memoria tra 1 executor (2GB) e 2 executor (4GB).

#### 3. Se vuole mantenere i numeri attuali per la relazione finale
Se per ragioni di tempo Matteo preferisce documentare i risultati attuali, digli che **può farlo purché li spieghi scientificamente** nel report:
> *"L'efficienza superlineare del 125.6% osservata a 2 core non è un'anomalia di calcolo, ma un tipico fenomeno di **Memory-Bound Super-linear Speedup** in ambiente distribuito: passando da 1 a 2 core in modalità Standalone con allocazione dinamica dei nodi, la memoria heap aggregata del cluster raddoppia da 2 GB a 4 GB. Sotto 1 core, il dataset pre-cachato causa saturazione della memoria ed elevata pressione sul Garbage Collector (GC time 11x superiore); a 2 core, il carico risiede interamente in RAM e parallelizza due processi worker, azzerando le contese di cache ed evidenziando una scalabilità superiore a quella puramente ideale."*