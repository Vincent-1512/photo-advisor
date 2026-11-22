"""Aggregate feedback from a local Photo Advisor SQLite snapshot."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def summarize(database: Path) -> dict:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        totals = connection.execute("""
            SELECT count(*) AS responses, count(DISTINCT f.job_id) AS jobs,
                   count(DISTINCT session_id) AS sessions,
                   count(preferred_over_original) AS comparisons,
                   count(CASE WHEN preferred_over_original = 1 THEN 1 END) AS preferred
            FROM feedback AS f JOIN jobs AS j USING (job_id) WHERE j.consent = 1
        """).fetchone()
        positions = connection.execute("""
            SELECT o.position + 1 AS position, count(*) AS impressions,
                   sum(CASE WHEN f.chosen_candidate_id = o.candidate_id THEN 1 ELSE 0 END) AS selections
            FROM feedback_shown_order AS o JOIN feedback AS f USING (feedback_id, job_id)
            JOIN jobs AS j USING (job_id) WHERE j.consent = 1
            GROUP BY o.position ORDER BY o.position
        """).fetchall()
        choices = connection.execute("""
            SELECT j.goal, coalesce(j.provider_type, 'baseline') AS provider,
                   coalesce(c.source, 'no_choice') AS chosen_source, count(*) AS responses
            FROM feedback AS f JOIN jobs AS j USING (job_id)
            LEFT JOIN candidates AS c ON c.job_id = f.job_id
                AND c.candidate_id = f.chosen_candidate_id
            WHERE j.consent = 1 GROUP BY j.goal, j.provider_type, c.source
            ORDER BY j.goal, provider, chosen_source
        """).fetchall()
    return {**dict(totals), "preference_share": (
        totals["preferred"] / totals["comparisons"] if totals["comparisons"] else None
    ), "choices": [dict(row) for row in choices], "positions": [dict(row) for row in positions]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    print(json.dumps(summarize(parser.parse_args().database), indent=2))
