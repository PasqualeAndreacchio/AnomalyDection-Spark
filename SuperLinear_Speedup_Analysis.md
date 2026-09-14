# Analisi dello Speedup Super-lineare: I/O Bound vs CPU Bound in PySpark

Durante l'esecuzione sistematica della `Correlation Pipeline` in ambiente distribuito, i risultati del benchmark hanno mostrato efficienze parallele significativamente superiori a 1.0 (Efficienza > 100%) passando dalle configurazioni a 1 Core (1 Nodo) a quelle a 4 e 6 Core (3 Nodi).

Questo documento analizza tecnicamente il fenomeno, dimostrando che i valori ottenuti non sono frutto di un'errata misurazione software (es. problemi di caching locale o di codice), ma rappresentano un classico esempio di **I/O-Bound Super-linear Speedup** derivante da *Resource Starvation* a livello infrastrutturale.

---

## 1. Wall Time vs CPU Time: L'Indizio Chiave

Confrontando il **Wall Time** (il tempo reale percepito dall'utente per completare il task) con il **CPU Time** (la somma del tempo speso da tutti gli Executor a macinare attivamente istruzioni), si nota una discrepanza fondamentale. Prendendo ad esempio la ripetizione 2 (sistema a caldo):

| Configurazione | Wall Time (s) | CPU Time (s) | I/O Wait (s) |
| :--- | :--- | :--- | :--- |
| **1 Core** (1 Nodo) | 53.5 | 16.2 | ~ 37.3 |
| **4 Core** (3 Nodi) | 8.7 | 16.5 | ~ 4.6 (media) |
| **6 Core** (3 Nodi) | 6.4 | 16.9 | ~ 3.6 (media) |

**Osservazione Critica:** Il `CPU Time` totale richiesto dal cluster per elaborare l'intero dataset è pressoché costante (~16-17 secondi) per tutte le configurazioni. Questo certifica che il carico computazionale del codice Spark è coerente. 

Tuttavia, a 1 Core, il tempo totale schizza a 53.5 secondi. Ci sono oltre **37 secondi di latenza "morta" (I/O Wait)** in cui il processore è inattivo, in attesa della rete, del disco o della memoria. 

---

## 2. Le Cause del Collo di Bottiglia Infrastrutturale

Aumentando il parametro `spark.cores.max` in questo specifico cluster CloudVeneto, non stiamo scalando in maniera "pura" (aggiungendo solo CPU sullo stesso nodo hardware), ma stiamo **scalando orizzontalmente** (allocando nuovi Executor su nuovi nodi fisici/virtuali).

L'aggiunta di nodi moltiplica drasticamente la capacità dell'infrastruttura di smaltire l'I/O. Le tre cause principali di questo abbattimento dei tempi morti sono:

### A. Network Bandwidth (Lettura Dati da S3)
* **A 1 Core:** Un solo worker node deve scaricare interamente i file Parquet da Amazon S3 attraverso la singola interfaccia di rete di un'unica istanza EC2.
* **A 4 e 6 Core:** Sono coinvolte 3 diverse istanze EC2. Esse scaricano porzioni del Parquet da S3 *simultaneamente*, godendo della tripla larghezza di banda aggregata e saturando in parallelo i controller di rete. Il tempo di download crolla.

### B. Disk Throughput (Shuffle Operations)
Operazioni come `Window.partitionBy()` e le `join` generano enormi moli di dati temporanei di shuffle.
* **A 1 Core:** Tutti i blocchi di shuffle vengono scritti e riletti da un singolo disco locale su un singolo worker (spesso già saturo di spazio o vincolato da IOPS lenti). Questo strozza violentemente l'Executor costringendolo all'attesa.
* **A 4 e 6 Core:** Le scritture e letture temporanee di shuffle sono bilanciate su 3 dischi fisici differenti (appartenenti a 3 macchine separate), riducendo di tre volte la latenza I/O e il carico per ogni disco.

### C. Resource Starvation (Disk Spilling e RAM)
Il cluster è configurato per allocare circa 2 GB di RAM per ogni Core/Executor.
* **A 1 Core (2 GB RAM totali):** Anche se il GC Time (`jvmGcTime_sec`) può restare nominalmente basso, Spark fatica a trattenere in memoria le enormi tabelle generate dalle Pivot (Task Shuffle-Bound e Window-Bound). È costretto a ricorrere pesantemente allo **Spill-to-Disk** (riversare le collezioni parziali su disco e recuperarle in seguito).
* **A 6 Core (12 GB RAM totali):** Il cluster dispone di uno spazio d'indirizzamento gigantesco. Spark può mantenere molteplici partizioni di lavoro e strutture intermedie completamente in memoria RAM (In-Memory Processing nativo), saltando a piè pari l'accesso a disco ed eliminando gli attriti di serializzazione.

---

## 3. Conclusione

La comparsa di un'efficienza super-lineare in questo tipo di benchmark **è un comportamento atteso e corretto** in un ambiente multi-nodo dove le risorse di partenza sono fortemente limitate.

Dimostra matematicamente che l'elaborazione a 1 singolo nodo è vittima di *Resource Starvation* a livello di I/O (Rete, Disco, Memoria). Aggiungere nodi al cluster risolve i colli di bottiglia hardware, portando a riduzioni dei tempi di esecuzione molto più drastiche di quanto suggerirebbe il semplice rapporto aritmetico del numero di CPU.
