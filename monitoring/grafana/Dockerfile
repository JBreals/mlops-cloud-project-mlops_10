FROM grafana/grafana:latest

USER root

# Install plugins
RUN grafana-cli plugins install grafana-piechart-panel
RUN grafana-cli plugins install grafana-worldmap-panel

# Copy provisioning and dashboards only
COPY provisioning/ /etc/grafana/provisioning/
COPY dashboards/ /var/lib/grafana/dashboards/

EXPOSE 3000