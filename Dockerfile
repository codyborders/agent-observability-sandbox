FROM python:3.11-slim

WORKDIR /app

ARG DD_GIT_REPOSITORY_URL=""
ARG DD_GIT_COMMIT_SHA=""
ARG DD_VERSION=""
ENV DD_GIT_REPOSITORY_URL=${DD_GIT_REPOSITORY_URL}
ENV DD_GIT_COMMIT_SHA=${DD_GIT_COMMIT_SHA}
ENV DD_VERSION=${DD_VERSION}

COPY requirements.txt ./
RUN pip install --no-cache-dir \
    --find-links=https://dd-trace-py-builds.s3.amazonaws.com/96035140/index.html \
    -r requirements.txt

COPY . .

EXPOSE 9000

RUN apt-get update && apt-get install -y cron curl && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
