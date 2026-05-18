CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    config_hash TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    total_items INTEGER,
    passed_items INTEGER,
    avg_composite_score REAL,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS eval_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES eval_runs(id),
    item_index INTEGER NOT NULL,
    input_text TEXT NOT NULL,
    reference_answer TEXT,
    model_response TEXT NOT NULL,
    semantic_score REAL,
    keyword_score REAL,
    structured_score REAL,
    composite_score REAL NOT NULL,
    passed INTEGER NOT NULL,
    latency_ms INTEGER,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_task ON eval_runs(task_name);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON eval_runs(provider);
