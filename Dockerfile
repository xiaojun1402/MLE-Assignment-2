# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps, bash, and other necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
        bash \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark's scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh 
    
# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Add Java module access flags for Spark compatibility with Java 17
ENV SPARK_SUBMIT_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"

# Set the working directory to Airflow's default
WORKDIR /opt/airflow

# Create necessary directories for the ML pipeline (as root for proper permissions)
RUN mkdir -p /opt/airflow/models \
    && mkdir -p /opt/airflow/monitoring_reports \
    && mkdir -p /opt/airflow/monitoring_workspace \
    && mkdir -p /opt/airflow/logs/ml_pipeline \
    && mkdir -p /opt/airflow/datamart/bronze \
    && mkdir -p /opt/airflow/datamart/silver \
    && mkdir -p /opt/airflow/datamart/gold \
    && mkdir -p /opt/airflow/datamart/gold/model_monitoring \
    && mkdir -p /opt/airflow/scripts \
    && mkdir -p /opt/airflow/scripts/utils \
    && mkdir -p /opt/airflow/utils \
    && mkdir -p /opt/airflow/data

# Copy the requirements file into the container
COPY requirements.txt ./

# Change ownership of directories to airflow user
RUN chown -R airflow:root /opt/airflow/models \
    && chown -R airflow:root /opt/airflow/monitoring_reports \
    && chown -R airflow:root /opt/airflow/monitoring_workspace \
    && chown -R airflow:root /opt/airflow/datamart \
    && chown -R airflow:root /opt/airflow/scripts \
    && chown -R airflow:root /opt/airflow/utils \
    && chown -R airflow:root /opt/airflow/data

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
# First install lighter packages
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Set proper working directory
WORKDIR /opt/airflow