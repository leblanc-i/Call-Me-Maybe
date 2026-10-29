*This project has been created as part of the 42 curriculum by <your_login>.*

# Call Me Maybe

## Description

Call Me Maybe translates natural-language requests into typed function calls.
It does not execute functions or compute their answers. Every successfully
written record contains exactly `prompt`, `name`, and `parameters`.

The implementation generates the complete call with schema-constrained,
token-by-token decoding through the public methods of the supplied SDK.
The default model is `Qwen/Qwen3-0.6B`.

**Before submission:** replace `<your_login>` in the first line with your actual
42 login. No login has been invented on your behalf.

## Instructions

Requirements: Python 3.10–3.13, a current `uv`, sufficient disk space and memory
for the SDK/model, and network access for the initial model download.

Run these commands from this directory:

```sh
uv sync
uv run python -m src
```

The default inputs are:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

The default output is `data/output/function_calls.json`, as required by
section V.4 of the subject. Choose another destination with `--output` when
needed.

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Optional controls:

```sh
uv run python -m src --device cpu --max-new-tokens 256
```

The SDK selects the device automatically when `--device` is omitted. The supplied
lock uses CPU PyTorch on Linux; CUDA cannot be enabled just by passing `--device
cuda` to a CPU-only installation. Keep the default on evaluation machines unless
you have deliberately prepared a compatible GPU environment.

Makefile targets: `install`, `run`, `debug`, `clean`, `lint`, `lint-strict`, `test`.
`lint` runs the exact commands and flags required by the subject. The provided
SDK is excluded from project lint/type checks, not altered to silence them.

## Local SDK packaging

The original SDK directory manifest and lock were not attached for this
adaptation. The archive reconstructs its packaging around the supplied
`init.py`, copied unchanged to `llm_sdk/llm_sdk/__init__.py`. A small public
re-export at `llm_sdk/__init__.py` supports imports from the repository root.

The SDK is a local uv workspace member. Its manifest, not the application,
declares PyTorch, Transformers, Hugging Face Hub and Accelerate. Application
code imports none of these libraries directly and accesses no SDK private
attributes. These are internal SDK dependencies, not a claim that the provided
SDK has no framework dependencies.

If you restore your original SDK manifest instead, inspect its package name and
dependencies and regenerate `uv.lock`. Do not blindly overwrite an official SDK
package with unrelated files. The included manifest is explicitly reconstructed,
not represented as the official supplied manifest.

## Algorithm explanation

1. Strict JSON loading rejects malformed data, non-standard constants and
   duplicate object keys. Pydantic validates the catalog and requests.
2. The prompt contains the current catalog and uses Qwen3's chat framing with
   an already closed thinking block. No private tokenizer is accessed.
3. `vocab.json` is loaded once. Qwen's byte-level BPE alphabet is inverted into
   actual bytes; raw symbols such as `Ġ` are not copied into generated text.
4. A byte-prefix trie indexes vocabulary tokens. An incremental grammar follows
   the complete call: fixed keys, a model-selected function name or an explicit
   null-name abstention, exact argument keys, typed values, and final braces.
5. For each step, the SDK returns next-token logits. A NumPy vector starts at
   negative infinity; only tokens whose *entire byte sequence* is legal retain
   their original score. The highest-scoring finite allowed token is chosen.
6. The chosen ID is appended to the actual model context. Grammar state advances
   over every byte, including tokens spanning several grammar states and UTF-8
   characters split over token boundaries.
7. Generation finishes only when the complete object has been produced. Numbers
   finish at grammar delimiters, never simply at the first valid digit.
8. Independent JSON and schema validation runs before a call is accepted. The
   complete batch replaces its destination atomically only after success.

Supported parameter/return descriptors are `number`, `string`, `boolean`,
`integer`, and `null`. Parameter objects/arrays and arbitrary JSON Schema
keywords are not implemented; they are rejected explicitly, not silently
accepted. The first three scalar types cover the supplied input catalog.

The grammar intentionally generates a canonical subset of valid JSON: fixed
key order and no insignificant whitespace outside string values. It preserves
spaces inside strings, supports empty strings and escaped characters, and
handles JSON fractions and exponents. Unknown names, extra/missing arguments
and wrong value types cannot be written as successful output records.

## Design decisions

- `src/model/llm.py`: one LLM wrapper owning one SDK instance and one decoder.
- `src/parser/`: input and output validation, no model operations.
- `src/prompt/`: catalog rendering and Qwen3 message framing.
- `src/constrained_decoding/grammar.py`: incremental grammar and lexical states.
- `src/constrained_decoding/vocabulary.py`: reversible byte mapping and trie.
- `src/constrained_decoding/decoding.py`: logits masking and generation budget.
- `src/__main__.py`: CLI, batch orchestration and atomic output writing.

All application data/service classes use Pydantic. `GenerationError` is a normal
exception subclass, not a Pydantic data container. The supplied external SDK is
kept unchanged. Broad exception handling is limited to application boundaries
where it ensures readable errors and a nonzero exit code.

An exhausted budget, unusable logits, invalid final value or SDK failure produces
an explicit error. No first-function fallback, guessed `0`, guessed `false` or
post-hoc repaired call is substituted. A failed batch writes no new partial
result; an older output file may remain unchanged and must not be mistaken for
a fresh successful run. If the model abstains because no catalog function fits,
the CLI reports the prompt index and the reason, then leaves the output file
unchanged. Successful output records retain the required three-field schema.
Always check the exit code.

## Performance analysis

Structural validation and semantic accuracy are different. A well-formed call
can still select the wrong function or extract the wrong argument.

The sandbox verified the grammar, masks, CLI error paths, typing, and real Qwen
vocabulary reconstruction. It did **not** load the real Qwen model. Therefore
neither the subject's 90%+ accuracy target nor its under-five-minute target has
been measured or certified. See `VALIDATION.txt` for exactly what was executed.

The vocabulary/trie is built once per LLM load. Allowed transitions are cached
within a generation. The SDK recomputes logits from the full context each time;
its public interface exposes no reusable KV cache, so this implementation does
not pretend to bypass that limitation. String-body masks allow many vocabulary
tokens; grammar traversal is not free. Measure the full workload on the actual
evaluation machine.

The CLI reports per-prompt time and total time including model loading to stderr.
Use a timer for the complete process:

```sh
time uv run python -m src
```

Review function selection and *all* argument values against a manually checked
expected file before claiming any accuracy percentage. Scripted unit-test logits
are not a benchmark of natural-language understanding.

## Challenges faced

- Multi-character tokens can cross JSON keys, values and delimiters: the grammar
  checks every byte, not just the last character.
- Unicode characters may span tokens: incomplete UTF-8 prefixes are retained
  instead of replaced by a replacement character.
- Numeric prefixes such as `2` may grow into `265` or `2.65`: stopping requires a
  following legal grammar delimiter.
- Function names may share prefixes: quoted-name completion distinguishes a
  short name from a longer one without longest-prefix fallback.
- JSON types must not coerce: booleans are explicitly excluded from numbers.

## Testing strategy

```sh
make lint
make lint-strict
make test
```

Normal tests require no model download. They cover multi-token numbers,
fractions/exponents, booleans, strings, Unicode/escapes, multi-character tokens,
function names sharing prefixes, empty parameters, duplicate input keys,
missing files, type mismatches, output atomicity, token-budget exhaustion,
unusable logits and schema changes.

One optional integration test checks the real vocabulary and tokenizer files:

```sh
QWEN_VOCAB=/absolute/path/vocab.json \
QWEN_TOKENIZER=/absolute/path/tokenizer.json \
uv run pytest -q
```

Without these environment variables, that test is skipped. It imports a
transitive tokenizer package only in the development test, never in `src`;
it does not replace the implementation's byte mapping or token selection.
The subject treats test programs as development aids, not graded deliverables.
Keep them for your verification and omit them from submission if required by
your local evaluation instructions.

## Resources

- Project subject, version 1.7: chapters IV, V, VI and VIII.
- Qwen3 model documentation: https://huggingface.co/Qwen/Qwen3-0.6B
- Python JSON: https://docs.python.org/3/library/json.html
- Pydantic models: https://docs.pydantic.dev/latest/concepts/models/
- uv workspaces: https://docs.astral.sh/uv/concepts/projects/workspaces/

AI assistance was used to audit the attached implementation, redesign the
incremental grammar and byte mapping, adapt application files and packaging,
create regression tests, and draft documentation. Real-model quality was not
measured. The student must review, understand, test and take responsibility for
all code before submission; this README does not assert that review has already
happened.
