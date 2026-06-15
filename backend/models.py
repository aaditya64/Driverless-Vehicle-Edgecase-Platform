from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(String, primary_key=True, default=generate_uuid)
    s3_key = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    uploader_id = Column(String, nullable=True)
    location_lat = Column(Float, nullable=True)
    location_lng = Column(Float, nullable=True)
    narrative = Column(String, nullable=True)
    status = Column(String, default="processing")  # processing, complete, failed


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, nullable=False, unique=True, index=True)
    display_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EditEvent(Base):
    __tablename__ = "edit_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    target = Column(String, nullable=False)
    before = Column(JSON, nullable=True)
    after = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Label(Base):
    __tablename__ = "labels"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    value = Column(String, nullable=False)  # safe, near_miss, collision
    source = Column(String, nullable=False)  # model, human
    confidence = Column(Float, nullable=True)

class Tag(Base):
    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    tag_type = Column(String, nullable=False)   # actor_type, road_type, weather etc.
    tag_value = Column(String, nullable=False)

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    text = Column(String, nullable=False)

class RiskTimeline(Base):
    __tablename__ = "risk_timelines"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    scores = Column(JSON, nullable=False)  # list of per-frame risk scores

class LabelChange(Base):
    __tablename__ = "label_changes"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    old_value = Column(String, nullable=False)
    new_value = Column(String, nullable=False)
    changed_by = Column(String, nullable=False)
    changed_at = Column(DateTime, default=datetime.utcnow)

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True, default=generate_uuid)
    incident_id = Column(String, ForeignKey("incidents.id"), nullable=False)
    annotator_id = Column(String, nullable=False)
    frame_start = Column(Float, nullable=False)
    frame_end = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    note = Column(String, nullable=True)
