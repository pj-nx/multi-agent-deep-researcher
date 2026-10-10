# Deep Researcher API image: works as an AWS Lambda container (via Lambda Web Adapter), on ECS
# Fargate / any container host (plain uvicorn), and for local docker-compose runs.
#
#   docker build -t deep-researcher .
#   docker run --rm -p 8000:8000 --env-file .env deep-researcher
#
# Embeddings default to the hosted provider here: a local model would add ~1 GB and a long cold start.
FROM public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 AS lambda-adapter

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    EMBEDDINGS_PROVIDER=openrouter CHECKPOINTER=postgres CHROMA_MODE=http \
    TRUST_PROXY_HEADERS=true PORT=8000
# Lambda Web Adapter: turns the Function URL invocation into an HTTP request to :8000 and streams
# the response. Harmless outside Lambda (the extension only activates under the Lambda runtime).
COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_INVOKE_MODE=response_stream AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/api/health

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY api ./api
COPY src ./src
COPY public ./public
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
