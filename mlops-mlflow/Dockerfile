FROM python:3.9-slim-buster

WORKDIR /app

# # APT 패키지 설치
# COPY requirements/apt.txt .
# RUN apt-get update && \
#     xargs -a apt.txt apt-get install -y && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements/python.txt .
RUN pip install --upgrade pip && \
    pip install -r python.txt

# 포트 노출 및 기본 명령어
EXPOSE 5050
CMD ["bash"]

