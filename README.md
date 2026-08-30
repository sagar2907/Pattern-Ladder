# Pattern Ladder

**Live: <https://pattern-ladder-1.streamlit.app/>**

A study engine for LeetCode practice. You describe what you are stuck on in
your own words — *"I keep failing problems where you shrink a window from the
left"* — and you get back the name of the pattern, an ordered set of problems
that drill it, and a stated reason for every recommendation.

The distinguishing idea: every LeetCode problem ships with a `similar_questions`
list. Across the corpus that is not metadata, it is a graph. Clustering it
yields problem *families* that LeetCode's own tags do not name; ordering a
family by difficulty turns a list of results into a study path.

```
query       "find the next greater element to the right"
understood  technique: monotonic stack | difficulty: any | mode: ramp

pattern     monotonic stack to find next greater or visible elements
            22 problems, discovered from the graph, not a LeetCode tag

your ladder
  Easy     496  Next Greater Element I        74% accepted   <- start here
  Medium   503  Next Greater Element II       66% accepted   follows Next Greater Element I
  Medium  2104  Sum of Subarray Ranges        60% accepted   follows Next Greater Element I
  Medium  1936  Add Minimum Number of Rungs   43% accepted
  Medium   556  Next Greater Element III      35% accepted   follows Next Greater Element I

where this leads
  Between / Array / Monotonic Stack                  13 problems
  two-pointer sliding window with variable length    24 problems
```

Nothing in that response is a keyword match, and every "follows" is a curated
link rather than an inference.

---

## How a query is answered

| Stage | What happens | Cost |
|---|---|---|
| **Understand** | The sentence becomes `{technique, difficulty, mode}`, shown back to you. One Groq call, or a deterministic offline parser. | ~0 (or one API call) |
| **Retrieve** | BM25 and a dense encoder each run over all 2,830 problems and are merged with Reciprocal Rank Fusion. | ~15 ms |
| **Rerank** | A cross-encoder rescores the top 50, reading each query–problem pair together. Its ordering is **fused with** the retrieval ordering, not substituted for it. | ~1.2 s |
| **Expand** | The surviving results are looked up in the graph; the family they agree on becomes the ladder, along with the patterns it leads on to. | <1 ms |
| **Explain** | A grounded one-line reason per result, assembled from retrieved fields only, plus the curated link that makes one rung follow another. | ~0 |

## Quickstart

```bash
uv sync --extra dev
uv run python scripts/build_index.py      # downloads the corpus, builds indexes (~2 min)
uv run streamlit run src/pattern_ladder/app.py
```

With an API key, `build_index.py --describe` additionally writes a one-line
description per family -- offline, once, cached. It turns "Monotonic Stack /
Stack" into "monotonic stack to find next greater or visible elements". No
query ever waits on it, and without a key the build produces the same artefacts
without that one field.

No API key is required. Without one the system uses the offline query parser
and says so in the interface. To enable the model path, copy `.env.example` to
`.env` and add a free [Groq](https://console.groq.com) key.

```bash
uv run pytest -q                  # the full suite, offline, no secrets
uv run ruff check .
uv run python scripts/evaluate.py --k 5 --verbose
```

### If the encoder fails to load on Windows

Windows 11 ships Smart App Control, and when it is enforced it refuses to load
unsigned native extension modules. SciPy -- pulled in through
sentence-transformers by way of scikit-learn -- is a set of exactly those. The
symptom is an `ImportError: DLL load failed ... An Application Control policy
has blocked this file`, naming a different `.pyd` on each run as the policy
works through them. `import scipy` succeeds; `import scipy.ndimage` does not.

This is a host policy rather than a fault in the project or in SciPy. Nothing
here can work around it, and it is worth being precise about what it does and
does not affect:

- the offline test suite still passes in full, because it substitutes a stub
  encoder and never reaches SciPy;
- Linux CI is unaffected, as is any container built from the `Dockerfile`;
- the hosted deployment is unaffected, because it runs on Linux.

Only local dense encoding on that machine breaks. Check the state with:

```powershell
Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -Name VerifiedAndReputablePolicyState
```

`1` is enforced, `2` is evaluation, `0` is off. Smart App Control can only be
turned off permanently -- re-enabling it requires reinstalling Windows -- so
the better options are to run the project under WSL2 or in the container,
either of which sidesteps the policy without weakening the host.

---

## Honest status

### Built and verified

Every number below was measured on this machine by a script in this repository,
and each is reproducible with the command named.

**Recommendation quality.** Rungs are ordered by how approachable a problem is
*and* how well regarded it is. The two are nearly independent (corpus
correlation +0.14), so acceptance rate alone is blind to whether a problem was
worth solving: *Design an Ordered Stream* is accepted 82% of the time and
approved by 13% of 4,115 voters, and used to lead its tier. Across all 137
families this moved poorly-regarded problems out of ladder heads, **59 to 40**.

**Corpus** (`scripts/build_index.py`)

| | |
|---|---|
| Rows in upstream dataset | 3,549 |
| Excluded: paid-only (empty statements) | 716 |
| Excluded: empty after HTML stripping | 3 |
| **Retrievable corpus** | **2,830** |

**The graph** — the project's central claim, measured rather than assumed

| | Curated links only | After backfill |
|---|---|---|
| Edges | 1,932 | 4,395 (1,932 curated + 2,463 inferred) |
| Problems with no edge | 1,074 (38.0%) | 298 (10.5%) |
| Connected components | 1,248 | 341 |
| Families (≥5 members) | 60 | **137** |
| Corpus inside a family | 49.2% | **86.3%** |
| Largest family | 63 | 42 |
| Name coherence | 0.790 | **0.845** |
| Agreement with LeetCode tags (NMI) | 0.618 | 0.667 |
| Families with a model-written description | - | 120 of 137 |
| Families sharing a name with another | - | 10 |
| Families with a recorded next-pattern link | - | 111 of 137 |

**Retrieval quality** (`scripts/evaluate.py`, 20 queries)

Both columns are real runs. The offline column is the deterministic parser with
no network; the live column adds one Groq call per query for query
understanding, and nothing else differs.

| Metric | Offline parser | Live model (`--live`) |
|---|---|---|
| hit@5 | 0.95 (19/20) | **1.00** (20/20) |
| hit@5, obliquely-phrased | **1.00** (6/6) | **1.00** (6/6) |
| hit@5, literal | 0.929 (13/14) | **1.00** (14/14) |
| Ladder contains the canonical problem | **0.85** | **0.85** |
| Ladder from the expected family | 0.833 | **0.917** |
| Headline names a technique not asked about | **0** of 12 | **0** of 12 |
| Mean rank of hits | **1.53** | 1.80 |
| Queries parsed by the model | 0 | **20/20**, no fallbacks |

The model earns its place on the query the offline parser missed: for "maximum
sum of a contiguous subarray" it recovers `dynamic programming` (Kadane's
algorithm), which no keyword in the sentence implies. It also recovers
techniques the rule parser has no cue for at all -- `fast-slow pointers`,
`interval merging`, `quickselect`. Mean rank is slightly worse because
expanding the query pulls in more plausible candidates; recall is perfect.

**Live API, measured over the 20 queries** (`openai/gpt-oss-20b`, free tier)

| | |
|---|---|
| Calls succeeded | **20/20** |
| Prompt tokens / query | 272 |
| Completion tokens / query | 100 |
| **Total / query** | **372** (max 474) |
| Model latency | 0.91 s mean, 1.48 s p95 |
| Throughput ceiling | **21.5 queries/min** (tokens/min binds before requests/min) |
| Daily ceiling | **538 queries/day** (tokens/day binds before the 1,000 requests/day) |
| Cost if metered | $0.05 per 1,000 queries |
| Median latency, warm | **1.2 - 1.6 s** (load-dependent; see note) |
| p95 latency, warm | 1.6 - 1.8 s |
| Cached index load | 0.06 s |
| Cold start to first answer | 20.8 s (model weights) |
| Resident memory, serving | 705 MB |
| **Peak memory, cold index build** | **712 MB** (was 976 before tuning the encode batch) |

Warm latency is quoted as a range because it is: repeated runs of the same
twenty queries on the same machine gave medians from 1.23 s to 1.58 s
depending on what else was running. The reranker is ~90% of it and is
CPU-bound, so a single figure would imply a precision the measurement does
not have.

**Engineering**: 236 tests passing with no network access and no secrets; ruff
clean; `uv.lock` and `.python-version` committed; the Docker image has been
rebuilt and run against the current commit, serving a real query from the
container under a hard 1 GB cap (2.44 GB image, peak RSS 862 MiB, index and
model weights baked in, artefacts byte-identical to the committed index); CI
runs lint and tests on Python 3.11 and
3.12 and fails if any test downloads a model; green on both, verified on the
current commit.

**On reproducibility**, precisely: the corpus, BM25 index, embeddings and
graph are byte-identical across runs. The *clustering* is reproducible only
against a pinned dependency set -- with an identical graph and a fixed seed,
Louvain returns 472 communities under numpy 2.5.2 and 473 under numpy 2.4.6.
The lockfile carries numpy twice, keyed on Python version, so `.python-version`
pins 3.12; without it a fresh clone resolving 3.11 gets a different numeric
stack and a slightly different family count. The numbers in this README are
from Python 3.12.

### Not done, and why

- **The live path is now verified** — see the table above — but running it
  exposed three defects that the offline suite could not have caught, because
  each degrades silently to the fallback and the fallback works well:
  `max_tokens` was too small for a reasoning model and failed **100%** of live
  calls; the default reasoning effort cost 40% more tokens than needed; and the
  system prompt made the model answer "single" to almost every query. All three
  are fixed and pinned by tests. The lesson is that a good fallback hides a
  broken primary path.
- **Deployed, and throttled on the first attempt.** It runs on Streamlit
  Community Cloud at <https://pattern-ladder-1.streamlit.app/> and returns
  correct results there, with the model path live: queries come back marked
  *parsed by the language model*. The first deployment was
  CPU-throttled by the host within a day: the index was treated as derived data
  and excluded from the repository, so every cold start rebuilt it -- roughly
  three minutes of sustained CPU to fetch the corpus and encode 2,830
  documents. Committing the built index (15 MB) removes that; a fresh clone now
  loads in **0.77s** with no build. The throttle expires on its own, and the
  fix takes effect when the app is next restarted. The brief originally
  targeted Hugging Face Spaces free CPU; that is no longer available, since HF
  now requires a paid plan for Gradio or Docker Spaces and no longer offers
  Streamlit as an SDK. Measured resident memory is **~700 MB**
  (`scripts/measure_memory.py`) against a ~1 GB ceiling, and that figure
  excludes the Streamlit server process, so the real headroom is smaller than
  it looks.
- **The image is large: 2.44 GB**, and 695 MB of that is PyTorch alone. That is
  the floor for a CPU transformer stack; the rest is the two checkpoints
  (184 MB), the index (15 MB), and about 2 MB of this project. Two things were
  wrong with it and are fixed. It rebuilt the index instead of copying the
  committed one, and a build inside the image has no API key, so it produced 137
  families with **zero** descriptions against the committed index's 120 -- three
  minutes spent making a worse artefact than the repository already held. And
  `uv sync` left its download cache in the image beside the virtualenv it had
  just copied into: 1,367 MB of duplicate wheels, a third of the image. The
  index step went from 192 s to 53 s and the image from 4.25 GB to 2.44 GB, with
  byte-identical behaviour. The README also shared a layer with the dependency
  manifest, so editing prose invalidated the install beneath it; moving it
  took a README-only rebuild from 150 s to 70 s. Nothing in the project builds
  the image -- not CI, not the deployment, which reads `uv.lock` directly -- so
  this cost is paid only when someone runs `docker build` by hand.
- **A family can hold two techniques, and one line cannot describe both.** The
  community tagged both Sliding Window and Prefix Sum holds *Minimum Size
  Subarray Sum* and *Range Sum Query 2D*. Its description is written once per
  family with no knowledge of any query, so it commits to one technique and is
  wrong for anyone who asked about the other. The interface now leads with the
  deterministic name in exactly that case, which stops the wrong pattern being
  named but does not split the family. Splitting it is the real fix and was not
  attempted: resolution was already swept against `family@1`, and raising it
  fragments families that are not mixed.

- **The evaluation set is 20 queries, written by one person who knew the
  system.** It is a tripwire against regressions, not a benchmark. `family@1`
  is scored on only 12 of them, so a single query moves it by 8 points; treat
  differences of that size as noise. No human relevance labels exist, so there
  is no measurement of whether the *ladders* are pedagogically good — only
  whether they contain the problem a reasonable person would expect.
- **The system cannot say "nothing matched."** Dense retrieval is exhaustive
  and always returns its nearest rows however distant, so a meaningless query
  produces confident-looking results. This is pinned by a test rather than
  hidden.
- **Ten of the 137 tag-derived names are not unique** -- four separate
  families are all called "Tree / Binary Tree". Building with `--describe`
  gives 120 of them a model-written description that distinguishes them
  ("post-order traversal to compute subtree aggregates"), but the remaining 17
  fall back to a name that may be ambiguous.
- **The model does not always parse a query the same way.** Repeating all
  twenty evaluation queries three times each at temperature 0, nineteen were
  bit-identical and one was not. Temperature 0 is greedy decoding, not a
  reproducibility guarantee. One observed run of the live evaluation scored
  0.95 rather than 1.00 for this reason.
- **Roughly 14% of the corpus sits in no family** and returns results without a
  ladder. Those are problems with no curated links and no sufficiently similar
  neighbour.

---


## Deploying

The repository is deployment-ready for Streamlit Community Cloud, which is the
free tier this targets. Three things were checked rather than assumed:

* **Dependency format.** Community Cloud reads `uv.lock` first, ahead of
  `requirements.txt` and the rest, so the committed lockfile is used directly
  and no second dependency file is needed. Adding one would in fact be harmful:
  only the first file found is processed.
* **CPU-only PyTorch.** The lockfile resolves `torch==2.13.0+cpu` from the
  PyTorch CPU index on Linux and contains **no NVIDIA packages at all**. The
  default PyPI build would pull the CUDA stack, which alone exceeds the memory
  ceiling.
* **The built index is committed, so a deployment never builds one.** This
  reverses an earlier decision and the reason is measured: with the index
  excluded as derived data, a hosted instance rebuilt it on every cold start --
  about three minutes of sustained CPU to download the corpus and encode 2,830
  documents. Streamlit Community Cloud throttles for exactly that, and did.
  Committing 15 MB makes a cold start a file read, and has a third benefit:
  the deployed app now serves the same clustering this README describes,
  model-written family descriptions included, rather than whatever its own
  environment produced.

  The bootstrap path remains for a working tree with no artefacts. Measured
  when it does run: **132 seconds**, peaking at **712 MB** against the ~1 GB
  ceiling. That peak was 976 MB before the encode batch was tuned, which left
  48 MB of headroom -- steady-state memory looked fine, and the peak is what a
  host kills you for.

To deploy, at [share.streamlit.io](https://share.streamlit.io):

1. Sign in with GitHub and authorise access to this repository.
2. New app, from this repo, branch `main`.
3. Set the main file path to `src/pattern_ladder/app.py`.
4. Under *Advanced settings*, set the **Python version to 3.12**. This is not
   cosmetic: the lockfile carries numpy twice, keyed on Python version, and the
   3.11 resolution produces a different numeric stack that clusters the graph
   into 138 families rather than the 137 documented here. Nothing breaks, but
   the numbers stop matching the ones in this file.
5. Optionally add a Groq key under *Advanced settings -> Secrets*, as TOML:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

   It must stay at the **root level**. Streamlit exposes only root-level
   secrets as environment variables, and this application reads
   `os.environ`, so a key nested under a `[section]` is silently ignored and
   the app quietly falls back to the offline parser. Without a key at all it
   runs on that parser anyway and labels every parse as such.

Nothing else is required. Because the index is committed, the app serves its
first query as soon as dependencies finish installing.

## Layout

```
src/pattern_ladder/
  config.py            every tuned constant, each with the measurement behind it
  data.py              corpus acquisition, filtering, normalisation
  text.py              HTML to plain text
  index/
    lexical.py         BM25 (bm25s)
    dense.py           embeddings, exact search
    build.py           the offline build; writes artifacts/ and the manifest
  retrieval/
    fusion.py          Reciprocal Rank Fusion
    rerank.py          cross-encoder
    search.py          the pipeline, family selection, ladder construction
  graph/
    build.py           link graph and the embedding backfill
    families.py        Louvain, naming, ladder ordering, evaluation metrics
    naming.py          optional model-written family descriptions, built offline
  understand/
    schema.py          the Intent type and its validation
    fallback.py        deterministic offline parser
    groq_client.py     the API call, and its guaranteed degradation
    cache.py           on-disk parse cache, keyed on query and prompt
  explain.py           grounded, non-generated result explanations
  engine.py            loads the artefacts and holds the query-time objects
  app.py               Streamlit interface

scripts/               build, evaluate, three parameter sweeps, an encoder
                       comparison, a memory probe, the documentation check,
                       and the PDF renderer
eval/                  the 20-query smoke set
docs/                  the report source and its PDF renderer
```

## Reproducing the parameter choices

None of the tuned constants are defaults or guesses; each was swept and the
losing configurations are recorded.

```bash
uv run python scripts/sweep_graph.py       # graph shape: coverage vs granularity
uv run python scripts/sweep_family.py      # graph params against ladder quality
uv run python scripts/sweep_retrieval.py   # title weighting, BM25 k1 and b
uv run python scripts/compare_encoders.py  # candidate dense encoders
```

## Data and models

| Component | Choice | Licence |
|---|---|---|
| Corpus | [`Alishohadaee/leetcode-problems-dataset`](https://huggingface.co/datasets/Alishohadaee/leetcode-problems-dataset) (`raw_data/leetcode_problems.json`) | MIT |
| Lexical | [`bm25s`](https://pypi.org/project/bm25s/) | MIT |
| Dense | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), 22.7M params | Apache-2.0 |
| Reranker | [`ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), 22.7M params | Apache-2.0 |
| Graph | `networkx` Louvain | BSD |
| Query understanding | Groq free tier, `openai/gpt-oss-20b` | — |

The dataset's default parquet configuration is an instruction-tuning set with
two columns and none of the structured fields this needs. The structured table
is only under `raw_data/`; `config.py` points there deliberately.

## Licence

MIT.
