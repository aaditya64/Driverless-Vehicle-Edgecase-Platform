from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

from database import get_db
from models import Incident, Label, LabelChange

app = FastAPI(title="Edge-Case Intelligence Platform")


# ── Schemas ───────────────────────────────────────────────────────────────────

class IncidentCreate(BaseModel):
    narrative: Optional[str] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None

class LabelOverride(BaseModel):
    value: str
    changed_by: str


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Incidents ─────────────────────────────────────────────────────────────────

@app.post("/incidents", status_code=201)
def create_incident(data: IncidentCreate, db: Session = Depends(get_db)):
    incident = Incident(
        id=str(uuid.uuid4()),
        s3_key="stub-key",
        narrative=data.narrative,
        location_lat=data.location_lat,
        location_lng=data.location_lng,
        status="processing",
        uploaded_at=datetime.utcnow()
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return {
        "id": incident.id,
        "status": incident.status,
        "narrative": incident.narrative,
        "location_lat": incident.location_lat,
        "location_lng": incident.location_lng,
        "uploaded_at": incident.uploaded_at
    }

@app.get("/incidents")
def list_incidents(
    label: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Incident)
    incidents = query.all()
    return {
        "incidents": [
            {
                "id": i.id,
                "status": i.status,
                "narrative": i.narrative,
                "location_lat": i.location_lat,
                "location_lng": i.location_lng,
                "uploaded_at": i.uploaded_at
            }
            for i in incidents
        ]
    }

@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {
        "id": incident.id,
        "status": incident.status,
        "narrative": incident.narrative,
        "location_lat": incident.location_lat,
        "location_lng": incident.location_lng,
        "uploaded_at": incident.uploaded_at
    }


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
    incidents = db.query(Incident).all()
    return {
        "incidents": [
            {
                "id": i.id,
                "status": i.status,
                "narrative": i.narrative,
                "location_lat": i.location_lat,
                "location_lng": i.location_lng,
                "uploaded_at": i.uploaded_at
            }
            for i in incidents
        ]
    }