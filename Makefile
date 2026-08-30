# Convenience targets. Everything here is also a plain command; nothing is
# hidden behind make that you could not run directly.

.PHONY: install index test lint fmt eval app sweep docs-check report clean

install:          ## Create the venv and install from the lockfile
	uv sync --extra dev

index:            ## Download the corpus and build all cached artefacts
	uv run python scripts/build_index.py

test:             ## Offline test suite: no network, no secrets
# No -q. pyproject sets it in addopts already, and a second one gives -qq,
# which suppresses the summary line the run is being read for.
	uv run pytest

lint:             ## Ruff, matching CI
	uv run ruff check .

fmt:              ## Apply the safe autofixes
	uv run ruff check --fix .

eval:             ## Smoke evaluation over eval/smoke_queries.json
	uv run python scripts/evaluate.py --k 5 --verbose

sweep:            ## Re-derive the graph parameters from scratch
	uv run python scripts/sweep_graph.py
	uv run python scripts/sweep_family.py

app:              ## Run the interface locally
	uv run streamlit run src/pattern_ladder/app.py

docs-check:       ## Verify the numbers quoted in the docs against the artefacts
	uv run python scripts/check_docs.py

report:           ## Render every document in docs/ to PDF
	uv run python scripts/render_pdf.py

clean:            ## Remove derived data. The committed index is left alone.
# git clean rather than rm -rf, because it will not touch tracked files.
# artifacts/ is no longer purely derived: the built index is committed, so
# `rm -rf artifacts` deletes fifteen megabytes of tracked content and leaves
# the working tree full of deletions. The claim that everything under here
# was rebuildable stopped being true the moment the index was committed, and
# this target was not revisited.
	git clean -xdf artifacts
	rm -rf .pytest_cache .ruff_cache
