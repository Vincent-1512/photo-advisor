# Photo Advisor backend

From repository root, install and run:

```sh
python -m pip install -r backend/requirements.txt -r requirements.txt
uvicorn backend.app.main:app --reload
```

SQLite and image artifacts default to `backend/data/`; set `PHOTO_ADVISOR_DB` and `PHOTO_ADVISOR_STORAGE` to override. Jobs run synchronously and pipeline outputs are stored beneath each job directory.

Run backend tests with `pytest backend/tests`.

Set PHOTO_ADVISOR_STYLE_COUNT to 4 (default) or 5. The API returns the original and up to that many distinct styles, capped at two for silhouette/memorial. Style generation runs locally on CPU. The CLI and Colab retain their existing diagnostic candidate lists.
