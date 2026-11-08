"""Synchronous FastAPI adapter for the existing Photo Advisor pipeline."""
import hashlib, io, json, shutil, sys, uuid, datetime
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response, Request
from fastapi.responses import Response as BinaryResponse
from PIL import Image
from . import db
from .models import *

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from photo_advisor.cli import run_pipeline

app=FastAPI(title="Photo Advisor API",version="1.0.0")
@app.exception_handler(HTTPException)
async def api_error_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return Response(content=json.dumps(exc.detail),status_code=exc.status_code,media_type="application/json")
    return Response(content=json.dumps({"error":{"code":"invalid_image","message":str(exc.detail)}}),status_code=exc.status_code,media_type="application/json")
MAX_IMAGE=15*1024*1024
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")
def error(code,msg,status): raise HTTPException(status_code=status,detail={"error":{"code":code,"message":msg}})
def stored(rel):
    p=(db.STORAGE/rel).resolve()
    if db.STORAGE.resolve() not in p.parents: raise HTTPException(404)
    return p
def get_job(job_id):
    with db.connect() as c:
        row=c.execute("select * from jobs where job_id=?",(job_id,)).fetchone()
        if not row: error("job_not_found","Không tìm thấy job.",404)
        cands=c.execute("select * from candidates where job_id=? order by created_at,candidate_id",(job_id,)).fetchall()
    result=None
    if row["status"]=="succeeded":
        result={"analysis":{"framing":"unknown","subject_description":"","background_description":"","lighting_condition":"mixed","uncertainty":"Pipeline hiện không cung cấp analysis chuẩn hóa."},"keep_original":{"recommended":False,"reason":"Compatibility field; not used for ranking."},"candidates":[],"retake_advice":None}
        for x in cands:
            result["candidates"].append(dict(candidate_id=x["candidate_id"],source=x["source"],style_id=x["style_id"],style_label=x["style_label"],option_id=x["option_id"],name=x["name"],rationale=x["rationale"],requested_recipe=json.loads(x["requested_recipe_json"]),effective_recipe=json.loads(x["effective_recipe_json"]),status=x["status"],fallback_reason=x["fallback_reason"],export_eligible=bool(x["export_eligible"]),preview_url=f"/v1/jobs/{job_id}/candidates/{x['candidate_id']}/preview"))
    e={"code":row["error_code"],"message":row["error_message"]} if row["error_code"] else None
    return {"job_id":job_id,"status":row["status"],"provider_type":row["provider_type"],"error":e,"result":result}

@app.get("/healthz")
def healthz(): return {"status":"ok"}

@app.post("/v1/jobs",status_code=202,response_model=JobAccepted)
async def create_job(image: UploadFile=File(...),goal: Goal=Form(...),consent: bool=Form(...)):
    if consent is not True: error("consent_required","Cần consent=true để xử lý ảnh.",400)
    payload=await image.read(MAX_IMAGE+1)
    if len(payload)>MAX_IMAGE: error("image_too_large","Kích thước ảnh vượt 15 MiB.",413)
    if image.content_type not in ("image/jpeg","image/png"): error("unsupported_format","Chỉ chấp nhận JPEG hoặc PNG.",415)
    try:
        with Image.open(io.BytesIO(payload)) as im:
            if im.format not in ("JPEG","PNG"): raise ValueError()
            im.verify()
    except Exception: error("invalid_image","Nội dung ảnh không hợp lệ.",422)
    job="job_"+uuid.uuid4().hex; ts=now(); rel=f"{job}/source{'.jpg' if image.content_type=='image/jpeg' else '.png'}"
    src=stored(rel); src.parent.mkdir(parents=True,exist_ok=True); src.write_bytes(payload)
    sha=hashlib.sha256(payload).hexdigest()
    with db.connect() as c: c.execute("insert into jobs values(?,?,?,?,?,?,?,?,?,?,?)",(job,"running",None,goal.value,1,rel,sha,None,None,ts,ts))
    try:
        out=stored(f"{job}/pipeline"); manifest=run_pipeline(payload,goal.value,str(out),input_file_path=str(src),input_source="user_file",provider_type="fixtures",run_id=job)
        meta=manifest.get("metadata",{}); provider=meta.get("vlm_summary",{}).get("provider_type")
        provider={"local_hf":"real","mock":"mock","replay":"replay"}.get(provider)
        with db.connect() as c:
            for n,item in enumerate(manifest.get("candidates",[])):
                source=item.get("source_type","rule_v0"); source="rule" if source.startswith("rule") else "noop" if source=="noop" else "vlm" if source.startswith("vlm") else "rule"
                cid="candidate_"+uuid.uuid4().hex[:12]; preview=item.get("preview_filename") or "preview_noop.png"; p=out/preview
                if not p.exists(): p=out/"preview_noop.png"
                preview_rel=f"{job}/pipeline/{p.name}"
                c.execute("insert into candidates values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(cid,job,source,None,None,item.get("option_id"),item.get("source_name") or source,item.get("notes") or "",json.dumps(item.get("requested_recipe") or {},allow_nan=False),json.dumps(item.get("effective_recipe") or {},allow_nan=False),item.get("status","success"),item.get("fallback_reason"),int(bool(item.get("export_eligible"))),preview_rel,item.get("output_sha256"),ts))
            c.execute("update jobs set status='succeeded',provider_type=?,updated_at=? where job_id=?",(provider,now(),job))
    except Exception:
        with db.connect() as c: c.execute("update jobs set status='failed',error_code='provider_unavailable',error_message='Pipeline hiện không khả dụng.',updated_at=? where job_id=?",(now(),job))
    return {"job_id":job,"status":"queued"}

@app.get("/v1/jobs/{job_id}",response_model=Job)
def read_job(job_id:str): return get_job(job_id)

@app.delete("/v1/jobs/{job_id}",status_code=204)
def delete_job(job_id:str):
    with db.connect() as c:
        row=c.execute("select 1 from jobs where job_id=?",(job_id,)).fetchone()
        if not row: error("job_not_found","Không tìm thấy job.",404)
        c.execute("delete from jobs where job_id=?",(job_id,))
    shutil.rmtree(stored(job_id),ignore_errors=True); return Response(status_code=204)

def candidate(job_id,candidate_id):
    with db.connect() as c: x=c.execute("select * from candidates where job_id=? and candidate_id=?",(job_id,candidate_id)).fetchone()
    if not x: error("job_not_found","Không tìm thấy job hoặc candidate.",404)
    return x
def jpeg_response(path):
    try:
        with Image.open(path) as im:
            b=io.BytesIO(); im.convert("RGB").save(b,"JPEG",quality=92); return BinaryResponse(b.getvalue(),media_type="image/jpeg")
    except Exception: error("job_not_found","Không tìm thấy ảnh.",404)
@app.get("/v1/jobs/{job_id}/candidates/{candidate_id}/preview")
def preview(job_id:str,candidate_id:str): return jpeg_response(stored(candidate(job_id,candidate_id)["preview_relative_path"]))
@app.post("/v1/jobs/{job_id}/candidates/{candidate_id}/export")
def export(job_id:str,candidate_id:str):
    x=candidate(job_id,candidate_id)
    if not x["export_eligible"]: raise HTTPException(409,detail={"error":{"code":"export_not_eligible","message":"Candidate không đủ điều kiện xuất ảnh."}})
    original=stored(x["preview_relative_path"]); dest=stored(f"{job_id}/exports/{candidate_id}.jpg"); dest.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(original) as im: im.convert("RGB").save(dest,"JPEG",quality=95)
    return jpeg_response(dest)

@app.post("/v1/jobs/{job_id}/feedback",status_code=204)
def feedback(job_id:str,payload:FeedbackInput):
    with db.connect() as c:
        if not c.execute("select 1 from jobs where job_id=?",(job_id,)).fetchone(): error("job_not_found","Không tìm thấy job.",404)
        cid=payload.chosen_candidate_id
        if cid and not c.execute("select 1 from candidates where job_id=? and candidate_id=?",(job_id,cid)).fetchone(): error("invalid_image","Candidate không thuộc job.",400)
        if payload.shown_order:
            allowed={r[0] for r in c.execute("select candidate_id from candidates where job_id=?",(job_id,))}
            if not set(payload.shown_order)<=allowed: error("invalid_image","shown_order chứa candidate không thuộc job.",400)
        fid="feedback_"+uuid.uuid4().hex
        c.execute("insert into feedback values(?,?,?,?,?,?,?)",(fid,job_id,cid,None if payload.preferred_over_original is None else int(payload.preferred_over_original),payload.reason,payload.session_id,now()))
        for i,cid in enumerate(payload.shown_order or []): c.execute("insert into feedback_shown_order values(?,?,?,?)",(fid,job_id,i,cid))
    return Response(status_code=204)
