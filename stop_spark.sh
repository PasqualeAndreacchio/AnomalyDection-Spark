cd $SPARK_HOME

./sbin/stop-all.sh

# Delete old spark jobs and cache
rm -rf /usr/local/spark/work/app-*
sudo rm -rf /opt/spark-tmp/*

# In caso ci siano problemi di connessione rifiutata eseguire il comando sotto
# pkill -f "spark.deploy"