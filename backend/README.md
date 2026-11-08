# Photo Advisor backend

From repository root, install and run:

```sh
python -m pip install -r backend/requirements.txt -r requirements.txt
uvicorn backend.app.main:app --reload
```

SQLite and image artifacts default to `backend/data/`; set `PHOTO_ADVISOR_DB` and `PHOTO_ADVISOR_STORAGE` to override. Jobs run synchronously and pipeline outputs are stored beneath each job directory.

Run backend tests with `pytest backend/tests`.
