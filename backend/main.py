from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime
import uuid

from database import get_db
from models import Incident, Label, LabelChange, Tag, Summary, RiskTimeline
from s3 import upload_video_to_s3, ensure_bucket_exists, get_presigned_video_url

app = FastAPI(title="Edge-Case Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    ensure_bucket_exists()

# ── Schemas ───────────────────────────────────────────────────────────────────

class LabelOverride(BaseModel):
    value: Literal["safe", "near_miss", "collision"]
    changed_by: str


def _serialize_label(label: Label | None) -> dict | None:
    if not label:
        return None
    return {
        "value": label.value,
        "source": label.source,
        "confidence": label.confidence,
    }


def _serialize_incident(
    incident: Incident,
    db: Session,
    *,
    include_ml: bool = False,
    include_video_url: bool = False,
) -> dict:
    label = db.query(Label).filter(Label.incident_id == incident.id).first()
    data = {
        "id": incident.id,
        "status": incident.status,
        "narrative": incident.narrative,
        "location_lat": incident.location_lat,
        "location_lng": incident.location_lng,
        "uploaded_at": incident.uploaded_at,
        "label": _serialize_label(label),
    }
    if include_ml:
        tags = db.query(Tag).filter(Tag.incident_id == incident.id).all()
        summary = db.query(Summary).filter(Summary.incident_id == incident.id).first()
        timeline = db.query(RiskTimeline).filter(RiskTimeline.incident_id == incident.id).first()
        data["tags"] = [{"tag_type": t.tag_type, "tag_value": t.tag_value} for t in tags]
        data["summary"] = summary.text if summary else None
        data["risk_timeline"] = timeline.scores if timeline else None
    if include_video_url:
        data["video_url"] = get_presigned_video_url(incident.s3_key)
    return data


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Incidents ─────────────────────────────────────────────────────────────────

@app.post("/incidents", status_code=201)
def create_incident(
    video_file: UploadFile = File(...),
    narrative: Optional[str] = Form(None),
    location_lat: Optional[float] = Form(None),
    location_lng: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    is_video = (
        video_file.content_type
        and (
            video_file.content_type.startswith("video/")
            or video_file.content_type == "application/octet-stream"
        )
    )
    if not is_video and video_file.filename:
        ext = video_file.filename.rsplit(".", 1)[-1].lower()
        is_video = ext in ("mp4", "webm", "mov", "avi", "mkv")
    if not is_video:
        raise HTTPException(status_code=400, detail="File must be a video.")

    try:
        s3_key = upload_video_to_s3(video_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

    incident = Incident(
        id=str(uuid.uuid4()),
        s3_key=s3_key,
        narrative=narrative,
        location_lat=location_lat,
        location_lng=location_lng,
        status="waiting",
        uploaded_at=datetime.utcnow()
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return _serialize_incident(incident, db)

@app.get("/incidents")
def list_incidents(
    label: Optional[str] = None,
    status: Optional[str] = None,
    sort: Literal["uploaded_at"] = "uploaded_at",
    order: Literal["asc", "desc"] = "desc",
    db: Session = Depends(get_db),
):
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    if label:
        query = query.join(Label, Label.incident_id == Incident.id).filter(Label.value == label)
    sort_col = Incident.uploaded_at
    query = query.order_by(sort_col.asc() if order == "asc" else sort_col.desc())
    incidents = query.all()
    return {
        "incidents": [_serialize_incident(i, db) for i in incidents],
    }

@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _serialize_incident(
        incident, db, include_ml=True, include_video_url=True
    )

@app.get("/incidents/{incident_id}/video")
def get_incident_video(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"video_url": get_presigned_video_url(incident.s3_key)}


# ── Labels ────────────────────────────────────────────────────────────────────

@app.patch("/incidents/{incident_id}/labels")
def override_label(incident_id: str, data: LabelOverride, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    existing_label = db.query(Label).filter(Label.incident_id == incident_id).first()

    change = LabelChange(
        id=str(uuid.uuid4()),
        incident_id=incident_id,
        old_value=existing_label.value if existing_label else "none",
        new_value=data.value,
        changed_by=data.changed_by,
        changed_at=datetime.utcnow()
    )
    db.add(change)

    if existing_label:
        existing_label.value = data.value
        existing_label.source = "human"
    else:
        new_label = Label(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            value=data.value,
            source="human",
            confidence=None
        )
        db.add(new_label)

    db.commit()
    return {
        "incident_id": incident_id,
        "new_value": data.value,
        "changed_by": data.changed_by,
        "changed_at": change.changed_at
    }


# ── Export ────────────────────────────────────────────────────────────────────

@app.get("/export")
def export_incidents(db: Session = Depends(get_db)):
    incidents = db.query(Incident).order_by(Incident.uploaded_at.desc()).all()
    return {
        "incidents": [
            _serialize_incident(i, db, include_ml=True) for i in incidents
        ],
    }