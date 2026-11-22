import io
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import main, db

def png_bytes():
    f=io.BytesIO(); Image.new("RGB",(420,420),(80,120,160)).save(f,"PNG"); return f.getvalue()

def setup_backend(tmp_path,monkeypatch):
    monkeypatch.setattr(db,"DB_PATH",tmp_path/"db.sqlite")
    monkeypatch.setattr(db,"STORAGE",tmp_path/"files")
    def fake_pipeline(data,goal,out_dir,**kwargs):
        out=Path(out_dir); out.mkdir(parents=True); Image.new("RGB",(420,420),(80,120,160)).save(out/"preview_noop.png")
        return {"metadata":{"vlm_summary":{"provider_type":"mock"}},"candidates":[{"candidate_id":"cand_0_noop","source_type":"noop","source_name":"Original","notes":"No edit","requested_recipe":{"pipeline_version":"pipeline_v0","exposure":0,"contrast":1,"saturation":1,"crop":None},"effective_recipe":{"pipeline_version":"pipeline_v0","exposure":0,"contrast":1,"saturation":1,"crop":None},"status":"success","export_eligible":True,"preview_filename":"preview_noop.png"}]}
    monkeypatch.setattr(main,"run_pipeline",fake_pipeline)

def test_health_and_sync_job_lifecycle(tmp_path,monkeypatch):
    setup_backend(tmp_path,monkeypatch); client=TestClient(main.app)
    assert client.get("/healthz").json()=={"status":"ok"}
    created=client.post("/v1/jobs",data={"goal":"natural_portrait","consent":"true"},files={"image":("a.png",png_bytes(),"image/png")})
    assert created.status_code==202; jid=created.json()["job_id"]
    job=client.get(f"/v1/jobs/{jid}"); assert job.status_code==200 and job.json()["status"]=="succeeded"
    cid=job.json()["result"]["candidates"][0]["candidate_id"]
    assert client.get(f"/v1/jobs/{jid}/candidates/{cid}/preview").headers["content-type"]=="image/jpeg"
    assert client.post(f"/v1/jobs/{jid}/candidates/{cid}/export").status_code==200
    fb={"chosen_candidate_id":cid,"preferred_over_original":True,"reason":"ok","session_id":"session_x","shown_order":[cid]}
    assert client.post(f"/v1/jobs/{jid}/feedback",json=fb).status_code==204
    with db.connect() as c: assert c.execute("select count(*) from feedback_shown_order").fetchone()[0]==1
    assert client.delete(f"/v1/jobs/{jid}").status_code==204
    assert client.get(f"/v1/jobs/{jid}").status_code==404

def test_upload_validation(tmp_path,monkeypatch):
    setup_backend(tmp_path,monkeypatch); c=TestClient(main.app)
    assert c.post("/v1/jobs",data={"goal":"natural_portrait","consent":"false"},files={"image":("a.png",png_bytes(),"image/png")}).status_code==400
    assert c.post("/v1/jobs",data={"goal":"natural_portrait","consent":"true"},files={"image":("a.webp",b"bad","image/webp")}).status_code==415


def test_pipeline_feedback_and_research_report(tmp_path, monkeypatch):
    from data_science.feedback_report import summarize

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "study.sqlite3")
    monkeypatch.setattr(db, "STORAGE", tmp_path / "images")
    client = TestClient(main.app)
    accepted = client.post("/v1/jobs", data={"goal": "vibrant", "consent": "true"},
                           files={"image": ("sample.png", png_bytes(), "image/png")})
    assert accepted.status_code == 202
    path = "/v1/jobs/" + accepted.json()["job_id"]
    job = client.get(path).json()
    assert job["status"] == "succeeded"
    candidates = job["result"]["candidates"]
    assert candidates[0]["source"] == "noop"
    assert 1 < len(candidates) <= 5
    assert len({c["style_id"] for c in candidates}) == len(candidates)
    chosen = next(c for c in candidates if c["source"] == "rule")
    order = [c["candidate_id"] for c in candidates]
    payload = {"session_id": "session_demo", "reason": "", "preferred_over_original": None,
               "chosen_candidate_id": chosen["candidate_id"], "shown_order": order}
    for invalid in ({"session_id": "invalid"}, {"shown_order": [order[0], order[0]]}):
        assert client.post(path + "/feedback", json={**payload, **invalid}).status_code == 422
    assert client.post(path + "/feedback", json=payload).status_code == 204
    report = summarize(db.DB_PATH)
    assert report["responses"] == 1 and report["preference_share"] is None
    assert sum(p["selections"] for p in report["positions"]) == 1
    assert sum(p["impressions"] for p in report["positions"]) == len(order)
    assert report["choices"][0]["chosen_source"] == "rule"
    assert client.delete(path).status_code == 204
    assert summarize(db.DB_PATH)["responses"] == 0
