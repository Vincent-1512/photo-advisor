-- Photo Advisor proposed schema, SQLite 3.35+ recommended.
-- Enable per connection: PRAGMA foreign_keys = ON;
-- Images/previews are never BLOBs here: store relative paths and SHA-256 only.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY CHECK (job_id GLOB 'job_*'),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    provider_type TEXT CHECK (provider_type IS NULL OR provider_type IN ('mock', 'replay', 'real')),
    goal TEXT NOT NULL CHECK (goal IN ('natural_portrait', 'vibrant', 'silhouette', 'full_body', 'keep_context', 'memorial')),
    consent INTEGER NOT NULL CHECK (consent = 1),
    image_relative_path TEXT NOT NULL,
    image_sha256 TEXT NOT NULL CHECK (length(image_sha256) = 64),
    error_code TEXT CHECK (error_code IS NULL OR error_code IN ('invalid_image', 'unsupported_format', 'image_too_large', 'consent_required', 'job_not_found', 'export_not_eligible', 'provider_unavailable')),
    error_message TEXT,
    created_at TEXT NOT NULL, -- ISO-8601 UTC, e.g. 2026-11-08T10:00:00Z
    updated_at TEXT NOT NULL,
    CHECK ((status = 'failed' AND error_code IS NOT NULL) OR status <> 'failed'),
    CHECK ((status = 'succeeded' AND error_code IS NULL) OR status <> 'succeeded')
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT NOT NULL CHECK (candidate_id GLOB 'candidate_*'),
    job_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('noop', 'rule', 'vlm', 'learned')),
    style_id TEXT CHECK (style_id IS NULL OR (length(style_id) > 6 AND style_id GLOB 'style_[a-z0-9_]*' AND style_id NOT GLOB '*[^a-z0-9_]*')),
    style_label TEXT CHECK (style_label IS NULL OR length(style_label) BETWEEN 1 AND 80),
    option_id TEXT CHECK (option_id IS NULL OR option_id IN ('option_1', 'option_2')),
    name TEXT NOT NULL,
    rationale TEXT NOT NULL,
    requested_recipe_json TEXT NOT NULL,
    effective_recipe_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'fallback_noop', 'rejected')),
    fallback_reason TEXT,
    export_eligible INTEGER NOT NULL CHECK (export_eligible IN (0, 1)),
    preview_relative_path TEXT NOT NULL,
    preview_sha256 TEXT CHECK (preview_sha256 IS NULL OR length(preview_sha256) = 64),
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
    CHECK ((source = 'noop' AND option_id IS NULL) OR source <> 'noop'),
    CHECK ((source = 'noop' AND style_id IS NULL AND style_label IS NULL) OR source <> 'noop')
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY CHECK (feedback_id GLOB 'feedback_*'),
    job_id TEXT NOT NULL,
    chosen_candidate_id TEXT,
    preferred_over_original INTEGER CHECK (preferred_over_original IS NULL OR preferred_over_original IN (0, 1)),
    reason TEXT NOT NULL CHECK (length(reason) <= 500),
    session_id TEXT NOT NULL CHECK (session_id GLOB 'session_*'),
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (job_id, chosen_candidate_id) REFERENCES candidates(job_id, candidate_id) ON DELETE CASCADE,
    UNIQUE (feedback_id, job_id)
);

-- The API carries shown_order as an ordered array. Store it normalized so
-- candidate membership and order are queryable without a JSON1 dependency.
CREATE TABLE IF NOT EXISTS feedback_shown_order (
    feedback_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0 AND position < 6),
    candidate_id TEXT NOT NULL,
    PRIMARY KEY (feedback_id, position),
    UNIQUE (feedback_id, candidate_id),
    FOREIGN KEY (feedback_id, job_id) REFERENCES feedback(feedback_id, job_id) ON DELETE CASCADE,
    FOREIGN KEY (job_id, candidate_id) REFERENCES candidates(job_id, candidate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_candidates_job ON candidates(job_id);
CREATE INDEX IF NOT EXISTS idx_feedback_job_created ON feedback(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_shown_candidate ON feedback_shown_order(job_id, candidate_id);

-- DELETE FROM jobs WHERE job_id = ? cascades rows in candidates, feedback,
-- and feedback_shown_order.
-- Backend must also remove the job's image/preview files from storage; SQLite
-- foreign keys cannot delete filesystem objects. Paths must be relative, validated
-- under the configured storage root, and never derived directly from user input.
