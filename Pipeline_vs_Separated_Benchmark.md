# Analisi: Pipeline Unificata vs Task Separati in PySpark

Questo documento analizza le ragioni per cui la scomposizione di una pipeline in micro-task in Apache Spark non può essere misurata accuratamente senza l'uso di cache intermedie, e perché l'approccio *end-to-end* (pipeline unificata) risulta preferibile per test di strong scaling.

---

## 1. Come funziona la Lazy Evaluation di Spark

In Spark, le trasformazioni (es. `withColumn`, `join`, `groupBy`) sono **lazy** e non vengono eseguite finché non viene invocata un'azione (come `.count()` o `.show()`). 

Se si prova a misurare le tre fasi di una pipeline senza l'ausilio di cache intermedie, il comportamento di Spark sarà il seguente:

1. **Misura Task 5A (Switch Counts):** 
   Invocando `df_anomaly1.count()`, Spark legge i dati dalla sorgente (es. S3), esegue l'aggregazione e calcola gli switch. Misuriamo questo tempo.
2. **Misura Task 5B (Group & Join):**
   Invocando `df_joined.count()`, poiché nulla è stato salvato in memoria (cache), Spark **non ricorda** i calcoli appena svolti. Il motore ripartirà dalla lettura S3, ricalcolerà l'aggregazione, ricalcolerà gli switch (Task 5A) e aggiungerà la join (Task 5B). 
3. **Misura Task 5C (Correlazione):**
   Invocando `df_correlation.count()`, Spark ripartirà per la terza volta da zero, ricalcolando l'intera catena: Task 5A + Task 5B + Task 5C.

### Perché i tempi non sono additivi?
Potrebbe sembrare logico isolare i tempi eseguendo una sottrazione (es. `Tempo(5B) = Tempo(A+B) - Tempo(A)`). Tuttavia, il **Catalyst Optimizer** di Spark valuta l'intero albero di esecuzione (DAG) e ottimizza dinamicamente le query, fondendo stage e calcoli (pipelining). Questo significa che eseguire (A+B) non costa necessariamente la somma esatta di A e B eseguiti singolarmente, rendendo inaffidabile qualsiasi deduzione matematica basata sulla sottrazione dei tempi di esecuzione.

---

## 2. L'uso delle Cache Intermedie nei Benchmark

Per misurare un sotto-task in maniera perfettamente isolata (come fatto in altri branch del progetto, ad esempio nei Task di Nicola), l'uso di `.cache()` è obbligatorio.

Il pattern corretto per isolare i tempi prevede:
1. Definire il **Task N**.
2. Eseguire `.cache()` sull'output del Task N.
3. Forzare la materializzazione in RAM invocando un'azione (es. `.count()`).
4. Passare il DataFrame cachato come input al **Task N+1**, in modo che la successiva azione misuri *solamente* la logica del Task N+1.

### I problemi riscontrati nell'Anomaly Detection 1
Nel tentativo iniziale di separare i task 5A, 5B e 5C, sono emersi due problemi legati proprio all'uso della cache:

1. **Il ciclo di ripetizione (Repeats):** Le chiamate a `.cache()` erano annidate all'interno del loop di benchmark senza essere accompagnate da un corrispondente `.unpersist()`. Al Repeat 1 il DAG veniva calcolato e parcheggiato in RAM. Al Repeat 2, Spark riconosceva che il DAG era già processato e rispondeva istantaneamente in frazioni di secondo, inquinando irreparabilmente le medie.
2. **Il collo di bottiglia a 1 Core (Memory/GC Thrashing):** A causa dei vincoli di configurazione del cluster, lavorare con 1 solo Core significava limitare la memoria totale a 2 GB. Il dataset intermedio cachato saturava questa soglia, scatenando continue chiamate al Garbage Collector (GC Time decuplicato) e rallentando drammaticamente il run a 1 Core. Passando a 2 Core (2 Nodi allocati = 4 GB totali), il dataset entrava in RAM senza sforzo, producendo uno speedup *superlineare* puramente illusorio (frutto del salto di memoria e non della reale potenza computazionale).

---

## 3. Conclusione: L'Approccio End-to-End

Se l'obiettivo primario è valutare la **scalabilità parallela (Speedup ed Efficienza)** di un cluster, l'approccio scientificamente più inattaccabile è misurare la **pipeline unificata (End-to-End)** senza cache intermedie o pre-caching.

In un sistema distribuito:
- Includere lettura dell'input, trasformazione, shuffle e computazione finale in un unico processo permette di testare la reale capacità di carico (Throughput) del cluster.
- L'overhead iniziale del Driver e della serializzazione dei task viene adeguatamente ammortizzato da un carico I/O e CPU prolungato.
- Rimuovendo i risultati parziali pre-cachati si annulla la disuguaglianza architetturale della memoria (2GB vs 4GB), garantendo che l'incremento di performance misurato derivi unicamente dall'aumento dei core di elaborazione, manifestando così un'efficienza coerente e *sublineare* (< 100%).
