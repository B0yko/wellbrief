# syntax=docker/dockerfile:1

# ---- builder: build the wheel and pre-generate the demo workspace -------------------------
# Runs on the build host's native architecture (BUILDPLATFORM) regardless of the final image's
# TARGETPLATFORM: the wheel is pure Python and the demo workspace it produces (SQLite database
# plus BM25/vector index files) is architecture-independent, so cross-compilation is never needed.
FROM --platform=$BUILDPLATFORM python:3.12-slim AS builder

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY evals ./evals

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist \
    && pip install --no-cache-dir /dist/*.whl

# Pre-generate the same workspace `wellbrief demo` would build on first run, so the final image
# starts from a warm cache: `demo`'s own reuse check (workspace name, seed, format and a content
# hash of the generated corpus, see `wellbrief.demo`) matches this one exactly, so the container's
# entrypoint finds it already built and skips straight to serving.
ENV WELLBRIEF_HOME=/opt/wellbrief-home
RUN python -c "\
from wellbrief.corpus.generator import SEED; \
from wellbrief.demo import WORKSPACE_NAME, prepare; \
from wellbrief.workspace import Workspace; \
prepare(Workspace.resolve(WORKSPACE_NAME), seed=SEED, formats='mixed', rebuild=True)"

# ---- final: minimal runtime image ----------------------------------------------------------
FROM python:3.12-slim

RUN groupadd --system wellbrief \
    && useradd --system --gid wellbrief --home-dir /home/wellbrief --create-home wellbrief

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl /root/.cache

COPY --from=builder /opt/wellbrief-home /opt/wellbrief-home
RUN chown -R wellbrief:wellbrief /opt/wellbrief-home

ENV WELLBRIEF_HOME=/opt/wellbrief-home
USER wellbrief
WORKDIR /home/wellbrief

EXPOSE 8765
ENTRYPOINT ["wellbrief"]
CMD ["demo", "--no-browser", "--host", "0.0.0.0"]
