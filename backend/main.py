from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import or_
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime
import json
import uuid

from database import get_db
from models import Annotation, Incident, Label, LabelChange, Tag, Summary, RiskTimeline
from ml_worker import start_ml_worker
from s3 import (
    delete_video_from_s3,
    upload_video_to_s3,
    ensure_bucket_exists,
    get_presigned_video_url,
)

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
    start_ml_worker()

# ── Schemas ───────────────────────────────────────────────────────────────────

class LabelOverride(BaseModel):
    value: Literal["safe", "near_miss", "collision"]
    changed_by: str


class TagItem(BaseModel):
    tag_type: str
    tag_value: str


class TagOverride(BaseModel):
    tags: list[TagItem]
    changed_by: str


def _serialize_label(label: Label | None) -> dict | None:
    if not label:
        return None
    return {
        "value": label.value,
        "source": label.source,
        "confidence": label.confidence,
    }


def _serialize_tags(db: Session, incident_id: str) -> list[dict]:
    tags = db.query(Tag).filter(Tag.incident_id == incident_id).all()
    return [{"tag_type": t.tag_type, "tag_value": t.tag_value} for t in tags]


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
        "tags": _serialize_tags(db, incident.id),
    }
    if include_ml:
        summary = db.query(Summary).filter(Summary.incident_id == incident.id).first()
        timeline = db.query(RiskTimeline).filter(RiskTimeline.incident_id == incident.id).first()
        data["summary"] = summary.text if summary else None
        data["risk_timeline"] = timeline.scores if timeline else None
    if include_video_url:
        data["video_url"] = get_presigned_video_url(incident.s3_key)
    return data


def _parse_context_tags(raw: Optional[str]) -> list[str]:
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if str(t).strip()]
    except json.JSONDecodeError:
        pass
    return [t.strip() for t in raw.split(",") if t.strip()]


def _save_context_tags(db: Session, incident_id: str, context_tags: list[str]) -> None:
    for value in context_tags:
        db.add(
            Tag(
                id=str(uuid.uuid4()),
                incident_id=incident_id,
                tag_type="context",
                tag_value=value,
            )
        )


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
    context_tags: Optional[str] = Form(None),
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
    db.flush()
    _save_context_tags(db, incident.id, _parse_context_tags(context_tags))
    db.commit()
    db.refresh(incident)
    return _serialize_incident(incident, db)

@app.get("/incidents")
def list_incidents(
    label: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tag_type: Optional[str] = None,
    tag_value: Optional[str] = None,
    has_location: Optional[bool] = None,
    sort: Literal["uploaded_at"] = "uploaded_at",
    order: Literal["asc", "desc"] = "desc",
    db: Session = Depends(get_db),
):
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    if label:
        query = query.join(Label, Label.incident_id == Incident.id).filter(Label.value == label)
    if q:
        pattern = f"%{q}%"
        query = query.filter(
            or_(Incident.narrative.ilike(pattern), Incident.id.ilike(pattern))
        )
    if date_from:
        query = query.filter(Incident.uploaded_at >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(Incident.uploaded_at <= datetime.fromisoformat(date_to))
    if tag_type or tag_value:
        query = query.join(Tag, Tag.incident_id == Incident.id)
        if tag_type:
            query = query.filter(Tag.tag_type == tag_type)
        if tag_value:
            query = query.filter(Tag.tag_value.ilike(f"%{tag_value}%"))
    if has_location is True:
        query = query.filter(
            Incident.location_lat.isnot(None),
            Incident.location_lng.isnot(None),
        )
    elif has_location is False:
        query = query.filter(
            or_(Incident.location_lat.is_(None), Incident.location_lng.is_(None))
        )
    sort_col = Incident.uploaded_at
    query = query.order_by(sort_col.asc() if order == "asc" else sort_col.desc())
    incidents = query.distinct().all()
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


@app.delete("/incidents/{incident_id}")
def delete_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        delete_video_from_s3(incident.s3_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 delete failed: {str(e)}")

    db.query(Annotation).filter(Annotation.incident_id == incident_id).delete()
    db.query(LabelChange).filter(LabelChange.incident_id == incident_id).delete()
    db.query(Label).filter(Label.incident_id == incident_id).delete()
    db.query(Tag).filter(Tag.incident_id == incident_id).delete()
    db.query(Summary).filter(Summary.incident_id == incident_id).delete()
    db.query(RiskTimeline).filter(RiskTimeline.incident_id == incident_id).delete()
    db.delete(incident)
    db.commit()
    return {"deleted": True, "incident_id": incident_id}


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


@app.patch("/incidents/{incident_id}/tags")
def override_tags(incident_id: str, data: TagOverride, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    db.query(Tag).filter(Tag.incident_id == incident_id).delete()
    for tag in data.tags:
        if not tag.tag_type.strip() or not tag.tag_value.strip():
            continue
        db.add(
            Tag(
                id=str(uuid.uuid4()),
                incident_id=incident_id,
                tag_type=tag.tag_type.strip(),
                tag_value=tag.tag_value.strip(),
            )
        )

    db.commit()
    return {
        "incident_id": incident_id,
        "tags": _serialize_tags(db, incident_id),
        "changed_by": data.changed_by,
        "changed_at": datetime.utcnow(),
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
