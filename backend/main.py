from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime
import json
import os
import uuid

from auth import (
    create_token,
    get_current_user,
    hash_password,
    user_response,
    verify_password,
)
from database import get_db
from models import (
    Annotation,
    EditEvent,
    Incident,
    Label,
    LabelChange,
    Tag,
    Summary,
    RiskTimeline,
    User,
)
from ml_worker import start_ml_worker
from s3 import (
    delete_video_from_s3,
    upload_video_to_s3,
    ensure_bucket_exists,
    get_presigned_video_url,
)

app = FastAPI(title="Edge-Case Intelligence Platform")


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS")
    if not raw:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
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
    changed_by: Optional[str] = None


class AuthPayload(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class TagItem(BaseModel):
    tag_type: str
    tag_value: str


class TagOverride(BaseModel):
    tags: list[TagItem]
    changed_by: Optional[str] = None


class SummaryOverride(BaseModel):
    text: str


def _labelize_tag_type(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _serialize_label(label: Label | None) -> dict | None:
    if not label:
        return None
    return {
        "value": label.value,
        "source": label.source,
        "confidence": label.confidence,
    }


def _serialize_user(user: User | None) -> dict | None:
    if not user:
        return None
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
    }


def _serialize_tags(db: Session, incident_id: str) -> list[dict]:
    tags = db.query(Tag).filter(Tag.incident_id == incident_id).all()
    return [{"tag_type": t.tag_type, "tag_value": t.tag_value} for t in tags]


def _serialize_edit_history(db: Session, incident_id: str) -> list[dict]:
    events = (
        db.query(EditEvent)
        .filter(EditEvent.incident_id == incident_id)
        .order_by(EditEvent.created_at.desc())
        .all()
    )
    user_ids = {event.user_id for event in events}
    users = (
        {user.id: user for user in db.query(User).filter(User.id.in_(user_ids)).all()}
        if user_ids
        else {}
    )
    return [
        {
            "id": event.id,
            "action": event.action,
            "target": event.target,
            "before": event.before,
            "after": event.after,
            "created_at": event.created_at,
            "user": _serialize_user(users.get(event.user_id)),
        }
        for event in events
    ]


def _serialize_incident(
    incident: Incident,
    db: Session,
    *,
    include_ml: bool = False,
    include_video_url: bool = False,
) -> dict:
    label = db.query(Label).filter(Label.incident_id == incident.id).first()
    uploader = (
        db.query(User).filter(User.id == incident.uploader_id).first()
        if incident.uploader_id
        else None
    )
    data = {
        "id": incident.id,
        "status": incident.status,
        "narrative": incident.narrative,
        "location_lat": incident.location_lat,
        "location_lng": incident.location_lng,
        "uploaded_at": incident.uploaded_at,
        "uploader": _serialize_user(uploader),
        "label": _serialize_label(label),
        "tags": _serialize_tags(db, incident.id),
    }
    if include_ml:
        summary = db.query(Summary).filter(Summary.incident_id == incident.id).first()
        timeline = db.query(RiskTimeline).filter(RiskTimeline.incident_id == incident.id).first()
        data["summary"] = summary.text if summary else None
        data["risk_timeline"] = timeline.scores if timeline else None
        data["edit_history"] = _serialize_edit_history(db, incident.id)
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


def _record_edit(
    db: Session,
    incident_id: str,
    user: User,
    action: str,
    target: str,
    before: object | None,
    after: object | None,
) -> None:
    db.add(
        EditEvent(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            user_id=user.id,
            action=action,
            target=target,
            before=before,
            after=after,
            created_at=datetime.utcnow(),
        )
    )


def _incident_query(
    db: Session,
    *,
    label: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tag_type: Optional[str] = None,
    tag_value: Optional[str] = None,
    has_location: Optional[bool] = None,
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
    return query


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/auth/signup", status_code=201)
def signup(payload: AuthPayload, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    password = payload.password
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email is already registered")

    display_name = payload.display_name.strip() if payload.display_name else email.split("@")[0]
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        display_name=display_name,
        password_hash=hash_password(password),
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"token": create_token(user), "user": user_response(user)}


@app.post("/auth/login")
def login(payload: AuthPayload, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"token": create_token(user), "user": user_response(user)}


@app.get("/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {"user": user_response(current_user)}


# ── Incidents ─────────────────────────────────────────────────────────────────

@app.post("/incidents", status_code=201)
def create_incident(
    video_file: UploadFile = File(...),
    narrative: Optional[str] = Form(None),
    location_lat: Optional[float] = Form(None),
    location_lng: Optional[float] = Form(None),
    context_tags: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
        uploaded_at=datetime.utcnow(),
        uploader_id=current_user.id,
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
    query = _incident_query(
        db,
        label=label,
        status=status,
        q=q,
        date_from=date_from,
        date_to=date_to,
        tag_type=tag_type,
        tag_value=tag_value,
        has_location=has_location,
    )
    sort_col = Incident.uploaded_at
    query = query.order_by(sort_col.asc() if order == "asc" else sort_col.desc())
    incidents = query.distinct().all()
    return {
        "incidents": [_serialize_incident(i, db) for i in incidents],
    }


@app.get("/tags/types")
def list_tag_types(db: Session = Depends(get_db)):
    rows = (
        db.query(Tag.tag_type, func.count(func.distinct(Tag.tag_value)))
        .group_by(Tag.tag_type)
        .order_by(Tag.tag_type.asc())
        .all()
    )
    return {
        "tag_types": [
            {
                "value": tag_type,
                "label": _labelize_tag_type(tag_type),
                "value_count": value_count,
                "has_value_options": value_count <= 50,
            }
            for tag_type, value_count in rows
        ],
    }


@app.get("/tags/values")
def list_tag_values(tag_type: str, db: Session = Depends(get_db)):
    rows = (
        db.query(Tag.tag_value)
        .filter(Tag.tag_type == tag_type)
        .distinct()
        .order_by(Tag.tag_value.asc())
        .all()
    )
    values = [value for (value,) in rows]
    has_value_options = len(values) <= 50 and all(len(value) <= 80 for value in values)
    return {
        "tag_type": tag_type,
        "values": values if has_value_options else [],
        "value_count": len(values),
        "has_value_options": has_value_options,
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
def delete_incident(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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
    _record_edit(
        db,
        incident_id,
        current_user,
        "delete",
        "incident",
        {"status": incident.status, "s3_key": incident.s3_key},
        None,
    )
    db.delete(incident)
    db.commit()
    return {"deleted": True, "incident_id": incident_id}


# ── Labels ────────────────────────────────────────────────────────────────────

@app.patch("/incidents/{incident_id}/labels")
def override_label(
    incident_id: str,
    data: LabelOverride,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    existing_label = db.query(Label).filter(Label.incident_id == incident_id).first()
    before = _serialize_label(existing_label)

    change = LabelChange(
        id=str(uuid.uuid4()),
        incident_id=incident_id,
        old_value=existing_label.value if existing_label else "none",
        new_value=data.value,
        changed_by=current_user.id,
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

    _record_edit(
        db,
        incident_id,
        current_user,
        "update",
        "label",
        before,
        {"value": data.value, "source": "human", "confidence": None},
    )
    db.commit()
    return {
        "incident_id": incident_id,
        "new_value": data.value,
        "changed_by": current_user.id,
        "changed_at": change.changed_at
    }


@app.patch("/incidents/{incident_id}/tags")
def override_tags(
    incident_id: str,
    data: TagOverride,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    before = _serialize_tags(db, incident_id)
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

    after = [{"tag_type": t.tag_type, "tag_value": t.tag_value} for t in data.tags]
    _record_edit(db, incident_id, current_user, "update", "tags", before, after)
    db.commit()
    return {
        "incident_id": incident_id,
        "tags": _serialize_tags(db, incident_id),
        "changed_by": current_user.id,
        "changed_at": datetime.utcnow(),
    }


@app.patch("/incidents/{incident_id}/summary")
def override_summary(
    incident_id: str,
    data: SummaryOverride,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Summary cannot be empty")

    summary = db.query(Summary).filter(Summary.incident_id == incident_id).first()
    before = {"text": summary.text} if summary else None
    if summary:
        summary.text = text
    else:
        summary = Summary(id=str(uuid.uuid4()), incident_id=incident_id, text=text)
        db.add(summary)
    incident.narrative = text
    _record_edit(db, incident_id, current_user, "update", "summary", before, {"text": text})
    db.commit()
    return {
        "incident_id": incident_id,
        "summary": text,
        "changed_by": current_user.id,
        "changed_at": datetime.utcnow(),
    }


# ── Export ────────────────────────────────────────────────────────────────────

@app.get("/export")
def export_incidents(
    label: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tag_type: Optional[str] = None,
    tag_value: Optional[str] = None,
    has_location: Optional[bool] = None,
    order: Literal["asc", "desc"] = "desc",
    db: Session = Depends(get_db),
):
    query = _incident_query(
        db,
        label=label,
        status=status,
        q=q,
        date_from=date_from,
        date_to=date_to,
        tag_type=tag_type,
        tag_value=tag_value,
        has_location=has_location,
    )
    sort_col = Incident.uploaded_at
    incidents = query.order_by(sort_col.asc() if order == "asc" else sort_col.desc()).distinct().all()
    return {
        "count": len(incidents),
        "incidents": [
            _serialize_incident(i, db, include_ml=True) for i in incidents
        ],
    }
