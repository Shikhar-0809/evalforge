\# EvalForge



A production-grade LLM evaluation framework. Define eval tasks as YAML configs, run them against multiple LLM providers, score responses across three dimensions, and generate HTML reports — all from a single CLI command.



Built as a portfolio project demonstrating async Python, multi-provider LLM integration, and config-driven evaluation design.



\---



\## What it does



\- \*\*Config-driven evals\*\*: Adding a new task requires only a YAML file — no Python

\- \*\*Multi-provider\*\*: Run the same task against Anthropic Claude, OpenAI GPT, Google Gemini, or local Ollama models

\- \*\*Three-dimensional scoring\*\*: Semantic similarity, keyword coverage, and structured correctness — combined into a weighted composite score

\- \*\*Versioned history\*\*: Every run is stamped with model, timestamp, and config hash — queryable for longitudinal analysis

\- \*\*HTML reports\*\*: Per-run breakdown of scores, pass rates, and top failures

\- \*\*FastAPI dashboard\*\*: Browse and export results via REST API



\---



\## Stack



\- \*\*Python 3.11+\*\* — async/await throughout

\- \*\*FastAPI\*\* — REST API and dashboard

\- \*\*SQLite + aiosqlite\*\* — local persistent storage, no ORM

\- \*\*sentence-transformers\*\* — local semantic similarity scoring

\- \*\*Typer + Rich\*\* — CLI with progress indicators

\- \*\*Jinja2\*\* — HTML report generation

\- \*\*Ollama\*\* — free local LLM inference (no API key needed)



\---



\## Quickstart



\### 1. Clone and set up environment



```bash

git clone https://github.com/Shikhar-0809/evalforge.git

cd evalforge

python -m venv venv

source venv/bin/activate  # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

```



\### 2. Install Ollama (free, local LLMs)



Download from https://ollama.com/download, then:



```bash

ollama pull gemma3:1b

```



\### 3. Run an eval



```bash

python -m evalforge run --task factual\_qa --provider ollama --model gemma3:1b

```



\### 4. Generate a report



```bash

python -m evalforge report --run-id <run-id-from-above>

```



Open `data/reports/<run-id>.html` in your browser.



\### 5. Start the dashboard



```bash

uvicorn evalforge.main:app --reload

```



Visit http://localhost:8000/docs



\---



\## Example results (gemma3:1b)



| Task | Pass Rate | Avg Score |

|------|-----------|-----------|

| factual\_qa | 10/10 (100%) | 0.94 |

| summarization | 5/5 (100%) | 0.80 |

| code\_explanation | 0/10 (0%) | 0.44 |

| json\_extraction | 0/10 (0%) | 0.16 |



Small models excel at factual recall and summarization but struggle with structured JSON output — exactly the kind of insight this framework is built to surface.



\---



\## Project structure
evalforge/
├── tasks/               # YAML task definitions
├── datasets/            # JSONL evaluation datasets
├── evalforge/
│   ├── llm/             # Provider adapters (Anthropic, OpenAI, Gemini, Ollama)
│   ├── scoring/         # Semantic, keyword, structured scorers
│   ├── runner/          # Async batch executor
│   ├── storage/         # SQLite persistence
│   ├── reports/         # Jinja2 HTML report generator
│   ├── api/             # FastAPI routers
│   └── cli.py           # Typer CLI

---

## CLI reference

```bash
# Run an eval
python -m evalforge run --task <name> --provider <provider> --model <model>

# List all runs
python -m evalforge list-runs

# List available tasks
python -m evalforge list-tasks

# Generate HTML report
python -m evalforge report --run-id <id>
```

---

## Adding a new task

Create `tasks/my_task.yaml`:

```yaml
name: my_task
version: "1.0"
description: "My custom eval task"

dataset:
  path: "datasets/my_task.jsonl"
  input_field: "input"
  reference_field: "reference"

prompt_template: |
  Answer the following question:
  {input}

scoring:
  semantic:
    weight: 0.5
    model: "all-MiniLM-L6-v2"
  keyword:
    weight: 0.3
    required_keywords: ["answer", "because"]
    min_coverage: 0.5
  structured:
    weight: 0.2
    type: "length_check"
    min_length: 20
    max_length: 500

pass_threshold: 0.65
max_concurrent: 5
```

Then run it — no Python required.

