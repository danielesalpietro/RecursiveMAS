FROM bitnami/spark:3.5

USER root

ARG LIVY_VERSION=0.8.0-incubating
ARG LIVY_MIRROR=https://archive.apache.org/dist/incubator/livy/${LIVY_VERSION}
ARG LIVY_ARCHIVE=apache-livy-${LIVY_VERSION}_2.12-bin.zip

RUN apt-get update && apt-get install -y --no-install-recommends wget unzip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget -q "${LIVY_MIRROR}/${LIVY_ARCHIVE}" -O livy.zip && \
    unzip -q livy.zip && \
    mv apache-livy-${LIVY_VERSION}_2.12-bin livy && \
    rm livy.zip && \
    mkdir -p /opt/livy/logs

COPY livy.conf /opt/livy/conf/livy.conf

ENV LIVY_HOME=/opt/livy
ENV SPARK_HOME=/opt/bitnami/spark
ENV PATH="${LIVY_HOME}/bin:${PATH}"
ENV JAVA_HOME=/opt/bitnami/java

EXPOSE 8998

HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
    CMD curl -sf http://localhost:8998/version || exit 1

CMD ["/opt/livy/bin/livy-server"]
