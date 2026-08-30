# Indexes and model weights are baked into the image at build time, not fetched
# on boot. That is the whole point: at request time the only work left is the
# cross-encoder pass over 50 candidates, so a cold container serves its first
# query without rebuilding a 2,830-document index or downloading 90MB of
# weights while a user waits.

FROM python:3.12-slim AS base

# The telemetry variable is set because transformers phones home on import
# unless told not to. Kept out of the line continuation below: a comment inside
# a continued instruction is not portable across Dockerfile parsers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_home \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first, as their own layer: application code changes far more
# often than the lockfile, and this keeps the expensive install cached.
#
# --no-install-project is required, not merely tidy, and it fails in a way
# worth spelling out because it is not the obvious one. `uv sync` also installs
# the root project, and hatchling is configured to build from src/, which this
# layer has not copied yet. It does not error: it happily builds and installs
# an *empty* pattern_ladder wheel. Copying the source afterwards does not
# repair that, because the install is a real one rather than editable, so the
# empty package keeps shadowing the real code and `import pattern_ladder`
# raises ModuleNotFoundError at the warm-up step below.
#
# Skipping the project here and syncing again after the source is present is
# what makes the dependency layer cacheable *and* correct.
COPY pyproject.toml uv.lock ./
# --no-cache matters more here than it looks. UV_LINK_MODE=copy above makes uv
# copy each wheel out of its cache into the virtualenv rather than hardlinking,
# so without this the image carries every dependency twice: measured at 1,367 MB
# of cache beside a 1,365 MB virtualenv, roughly a third of the whole image,
# none of it reachable at runtime. The layer stays cacheable either way, because
# that is Docker's cache and not uv's.
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# README.md is needed to build the project wheel -- pyproject sets
# readme = "README.md" and hatchling reads it -- but it is deliberately not
# copied with pyproject.toml above. Sharing a layer with the dependency
# manifest meant every edit to the README invalidated the dependency install
# and the model warm-up beneath it: a documentation typo cost a hundred
# seconds of rebuild. It belongs with the source, which is what it describes
# and what changes at the same time.
COPY README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

# Now the project itself, against the dependency layer above.
RUN uv sync --frozen --no-dev --no-cache

# The built index is copied in rather than rebuilt. It is committed to the
# repository, so rebuilding it here spent three minutes of build time producing
# a *worse* artefact: a build inside the image has no API key, so it writes 137
# families with zero descriptions where the committed index has 120. The image
# was quietly a degraded version of the deployed application, and both manifests
# said so all along -- described_by_model 0 against 120 -- but nothing compared
# them.
COPY artifacts/index/ ./artifacts/index/

# Loading the index here rather than only at runtime makes a bad copy fail the
# build instead of the first query. Warming the model cache is in the same layer
# because an index is worthless without the weights it was built against.
RUN uv run python -c "from pattern_ladder.engine import load_engine; load_engine()" \
    && uv run python -c "from pattern_ladder.index.dense import get_encoder; from pattern_ladder.retrieval.rerank import get_reranker; get_encoder(); get_reranker()"

EXPOSE 8501

# Streamlit binds to localhost by default, which is unreachable from outside
# the container.
CMD ["uv", "run", "streamlit", "run", "src/pattern_ladder/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
