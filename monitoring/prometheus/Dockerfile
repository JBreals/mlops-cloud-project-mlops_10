FROM prom/prometheus:latest

USER root

# Copy configuration files
COPY prometheus.yml /etc/prometheus/prometheus.yml
COPY rules/ /etc/prometheus/rules/

EXPOSE 9090