FROM prom/alertmanager:latest

# Copy configuration
COPY alertmanager.yml /etc/alertmanager/

# Use numeric user ID instead of username
USER 1000:1000

EXPOSE 9093