FROM apache/airflow:latest-python3.12

ADD requirements.txt .
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    heimdal-dev \
    procps \
    ant \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Hadoop configurations
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
RUN mkdir -p $HADOOP_HOME/etc/hadoop

# Copy core-site.xml and hdfs-site.xml
COPY hadoop-conf/core-site.xml $HADOOP_CONF_DIR/
COPY hadoop-conf/hdfs-site.xml $HADOOP_CONF_DIR/

USER airflow
RUN pip install -r requirements.txt

CMD ["webserver"]