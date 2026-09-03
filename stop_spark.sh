cd $SPARK_HOME

./sbin/stop-all.sh

# Delete old spark jobs and cache
rm -rf /usr/local/spark/work/app-*
sudo rm -rf /opt/spark-tmp/*

for slave in slave-1 slave-2 slave-3; do
  echo "Emptying $slave..."
  ssh $slave "rm -rf /usr/local/spark/work/*"          # old Spark jobs
  ssh $slave "rm -rf /usr/local/spark/logs/*.out"      # Spark logs
  ssh $slave "rm -rf ~/.ivy2.5.2/jars/* ~/.ivy2.5.2/cache/*"   # cache JAR Ivy 
  ssh $slave "sudo rm -rf /tmp/spark-*"                # tmp Spark
done

# In caso ci siano problemi di connessione rifiutata eseguire il comando sotto
# pkill -f "spark.deploy"