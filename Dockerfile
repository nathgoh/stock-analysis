FROM flink:2.2.0-scala_2.12-java21

# Download the connector libraries
USER root
RUN mkdir -p /opt/sql-client/lib && \
    wget -P /opt/sql-client/lib/ https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/5.0.0-2.2/flink-sql-connector-kafka-5.0.0-2.2.jar && \
    wget -P /opt/sql-client/lib/ https://repo.maven.apache.org/maven2/org/apache/flink/flink-json/2.2.0/flink-json-2.2.0.jar && \
    chown -R flink:flink /opt/sql-client
USER flink
