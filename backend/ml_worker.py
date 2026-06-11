import base64
import json
import logging
import os
import ssl
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from sqlalchemy.orm import Session

from database import SessionLocal
from models import Incident, Label, RiskTimeline, Summary, Tag
from s3 import get_presigned_video_url

logger = logging.getLogger(__name__)

ML_API_URL = os.getenv("ML_API_URL")
ML_API_TIMEOUT_SECONDS = int(os.getenv("ML_API_TIMEOUT_SECONDS", "300"))
ML_WORKER_POLL_SECONDS = int(os.getenv("ML_WORKER_POLL_SECONDS", "5"))
ML_API_VERIFY_SSL = os.getenv("ML_API_VERIFY_SSL", "0").lower() in {"1", "true", "yes"}
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ML_API_VIDEO_SOURCE = os.getenv(
    "ML_API_VIDEO_SOURCE",
    "github" if GITHUB_REPO and GITHUB_TOKEN else "s3",
).lower()
_configured_github_mirror = os.getenv("GITHUB_MIRROR") or os.getenv("ML_API_GITHUB_MIRROR")
if ML_API_VIDEO_SOURCE == "github" and _configured_github_mirror in (None, "", "direct", "none", "0"):
    ML_API_GITHUB_MIRROR = "https://gh-proxy.com/"
else:
    ML_API_GITHUB_MIRROR = _configured_github_mirror or "direct"

_worker_started = False
_worker_lock = threading.Lock()


def start_ml_worker() -> None:
    global _worker_started
    if not ML_API_URL:
        logger.warning("ML_API_URL is not set; incident analysis worker is disabled.")
        return

    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_worker_loop, name="ml-analysis-worker", daemon=True)
        thread.start()
        _worker_started = True
        logger.info("ML analysis worker started.")


def _worker_loop() -> None:
    while True:
        try:
            _process_next_waiting_incident()
        except Exception:
            logger.exception("ML worker loop failed.")
        time.sleep(ML_WORKER_POLL_SECONDS)


def _process_next_waiting_incident() -> None:
    db = SessionLocal()
    try:
        incident = (
            db.query(Incident)
            .filter(Incident.status == "waiting")
            .order_by(Incident.uploaded_at.asc())
            .first()
        )
        if not incident:
            return

        incident.status = "processing"
        db.commit()
        incident_id = incident.id
        logger.info("Processing incident %s", incident_id)
    finally:
        db.close()

    db = SessionLocal()
    try:
        incident = db.query(Incident).filter(Incident.id == incident_id).first()
        if not incident:
            return

        analysis = _call_ml_api(incident)
        _save_analysis(db, incident, analysis)
        incident.status = "completed"
        db.commit()
        logger.info("Completed incident %s", incident.id)
    except Exception:
        db.rollback()
        failed = db.query(Incident).filter(Incident.id == incident_id).first()
        if failed:
            failed.status = "failed"
            db.commit()
        logger.exception("Failed to process incident %s", incident_id)
    finally:
        db.close()


def _call_ml_api(incident: Incident) -> dict[str, Any]:
    video_url = get_presigned_video_url(incident.s3_key, expires_in=ML_API_TIMEOUT_SECONDS + 600)
    filename = Path(incident.s3_key).name

    if ML_API_VIDEO_SOURCE == "github":
        relay = _upload_presigned_video_to_github(video_url, filename)
        try:
            return _post_ml_api(
                video_url=relay["download_url"],
                filename=relay["filename"],
                github_mirror=ML_API_GITHUB_MIRROR,
            )
        finally:
            _delete_github_asset(relay["path"], relay["sha"])

    return _post_ml_api(
        video_url=video_url,
        filename=filename,
        github_mirror=ML_API_GITHUB_MIRROR,
    )


def _post_ml_api(video_url: str, filename: str, github_mirror: str) -> dict[str, Any]:
    payload = {
        "video_url": video_url,
        "filename": filename,
        "github_mirror": github_mirror,
    }
    body = json.dumps(payload).encode("utf-8")
    api_request = request.Request(
        ML_API_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    context = None if ML_API_VERIFY_SSL else ssl._create_unverified_context()
    try:
        with request.urlopen(api_request, timeout=ML_API_TIMEOUT_SECONDS, context=context) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"ML API returned HTTP {exc.code}: {raw_error[:1000]}") from exc

    data = json.loads(raw)
    if data.get("status") != "completed":
        raise RuntimeError(f"ML API did not complete analysis: {data.get('status')}")
    return data


def _upload_presigned_video_to_github(video_url: str, filename: str) -> dict[str, str]:
    if not GITHUB_REPO or not GITHUB_TOKEN:
        raise RuntimeError(
            "ML_API_VIDEO_SOURCE=github requires GITHUB_REPO and GITHUB_TOKEN."
        )

    clean_filename = Path(filename).name
    if not clean_filename.lower().endswith(".mp4"):
        clean_filename = f"{clean_filename}.mp4"
    remote_path = f"relay/{int(time.time())}_{clean_filename}"

    logger.info("Downloading presigned S3 video for GitHub relay.")
    with request.urlopen(video_url, timeout=120) as response:
        video_bytes = response.read()
    logger.info("Uploading %s bytes to GitHub relay path %s.", len(video_bytes), remote_path)

    repo_info = _github_json(f"https://api.github.com/repos/{GITHUB_REPO}")
    branch = repo_info["default_branch"]
    payload = {
        "message": f"upload {remote_path}",
        "content": base64.b64encode(video_bytes).decode("utf-8"),
        "branch": branch,
    }
    upload = _github_json(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/{parse.quote(remote_path)}",
        data=payload,
        method="PUT",
        timeout=120,
    )
    return {
        "path": remote_path,
        "sha": upload["content"]["sha"],
        "download_url": upload["content"]["download_url"],
        "filename": Path(remote_path).name,
    }


def _delete_github_asset(path: str, sha: str) -> None:
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return
    try:
        _github_json(
            f"https://api.github.com/repos/{GITHUB_REPO}/contents/{parse.quote(path)}",
            data={"message": f"delete {path}", "sha": sha},
            method="DELETE",
            timeout=30,
        )
        logger.info("Deleted GitHub relay path %s.", path)
    except Exception:
        logger.exception("Failed to delete GitHub relay path %s.", path)


def _github_json(
    url: str,
    *,
    data: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: int = 30,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }
    encoded = json.dumps(data).encode("utf-8") if data is not None else None
    github_request = request.Request(url, data=encoded, method=method, headers=headers)
    try:
        with request.urlopen(github_request, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"GitHub returned HTTP {exc.code}: {raw_error[:1000]}") from exc


def _save_analysis(db: Session, incident: Incident, data: dict[str, Any]) -> None:
    briefing = _dict(data.get("briefing"))
    outcome = _dict(data.get("outcome"))
    risk = _dict(data.get("risk"))

    summary_text = _first_text(
        briefing.get("summary"),
        briefing.get("timeline"),
    )
    if summary_text:
        _upsert_summary(db, incident.id, summary_text)
        if not incident.narrative:
            incident.narrative = summary_text

    label_value = str(outcome.get("label") or "").strip()
    if label_value:
        _upsert_model_label(
            db,
            incident.id,
            label_value,
            str(outcome.get("source") or "model"),
            _label_confidence(outcome, label_value),
        )

    timeline = risk.get("timeline")
    if isinstance(timeline, dict) and isinstance(timeline.get("points"), list):
        _upsert_risk_timeline(db, incident.id, timeline)

    for tag_type, tag_value in _analysis_tags(data):
        _add_tag_once(db, incident.id, tag_type, tag_value)


def _upsert_summary(db: Session, incident_id: str, text: str) -> None:
    summary = db.query(Summary).filter(Summary.incident_id == incident_id).first()
    if summary:
        summary.text = text
        return
    db.add(Summary(id=str(uuid.uuid4()), incident_id=incident_id, text=text))


def _upsert_risk_timeline(
    db: Session,
    incident_id: str,
    timeline: dict[str, Any],
) -> None:
    existing = db.query(RiskTimeline).filter(RiskTimeline.incident_id == incident_id).first()
    if existing:
        existing.scores = timeline
        return
    db.add(
        RiskTimeline(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            scores=timeline,
        )
    )


def _upsert_model_label(
    db: Session,
    incident_id: str,
    value: str,
    source: str,
    confidence: float | None,
) -> None:
    label = db.query(Label).filter(Label.incident_id == incident_id).first()
    if label and label.source == "human":
        return
    if label:
        label.value = value
        label.source = source
        label.confidence = confidence
        return
    db.add(
        Label(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            value=value,
            source=source,
            confidence=confidence,
        )
    )


def _add_tag_once(db: Session, incident_id: str, tag_type: str, tag_value: str) -> None:
    tag_type = tag_type.strip()
    tag_value = tag_value.strip()
    if not tag_type or not tag_value:
        return
    existing = (
        db.query(Tag)
        .filter(
            Tag.incident_id == incident_id,
            Tag.tag_type == tag_type,
            Tag.tag_value == tag_value,
        )
        .first()
    )
    if existing:
        return
    db.add(
        Tag(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            tag_type=tag_type,
            tag_value=tag_value,
        )
    )


def _analysis_tags(data: dict[str, Any]) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    classification = _dict(data.get("classification"))
    risk = _dict(data.get("risk"))
    actor = _dict(data.get("actor"))
    environment = _dict(data.get("environment"))
    behavior = _dict(data.get("behavior"))
    collision = _dict(data.get("collision"))
    near_miss = _dict(data.get("near_miss"))
    review = _dict(data.get("review"))
    runtime = _dict(data.get("runtime"))
    outcome = _dict(data.get("outcome"))
    briefing = _dict(data.get("briefing"))

    _append(tags, "schema_version", data.get("schema_version"))
    _append(tags, "clip_id", data.get("clip_id"))
    _append(tags, "outcome_label", outcome.get("label"))
    _append(tags, "outcome_source", outcome.get("source"))
    _append(tags, "collision_score", outcome.get("collision_score"))
    _append(tags, "near_miss_score", outcome.get("near_miss_score"))
    _append(tags, "threshold", outcome.get("threshold"))
    _append(tags, "classification_primary_type", classification.get("primary_type"))
    _append(tags, "classification_confidence", classification.get("confidence"))
    _append_many(tags, "scenario", classification.get("scenario_tags"))
    _append_many(tags, "type", classification.get("type_tags"))
    _append(tags, "risk_level", risk.get("level"))
    _append(tags, "ego_relevance", risk.get("ego_relevance"))
    risk_timeline = _dict(risk.get("timeline"))
    peak = _dict(risk_timeline.get("peak"))
    _append(tags, "timeline_source", risk_timeline.get("source"))
    _append(tags, "timeline_score_type", risk_timeline.get("score_type"))
    _append(tags, "timeline_resolution", risk_timeline.get("temporal_resolution"))
    _append(tags, "timeline_frame_count", risk_timeline.get("frame_count"))
    _append(tags, "timeline_fps", risk_timeline.get("fps"))
    _append(tags, "timeline_duration_sec", risk_timeline.get("duration_sec"))
    _append(tags, "timeline_peak_frame", peak.get("frame_idx"))
    _append(tags, "timeline_peak_time_sec", peak.get("time_sec"))
    _append(tags, "timeline_peak_risk_score", peak.get("risk_score"))
    _append(tags, "actor_type", actor.get("type"))
    _append(tags, "actor_position", actor.get("position"))
    _append(tags, "lighting", environment.get("lighting"))
    _append(tags, "weather", environment.get("weather"))
    _append(tags, "road_surface", environment.get("road_surface"))
    _append(tags, "road_type", environment.get("road_type"))
    _append_many(tags, "visibility_issue", environment.get("visibility_issues"))
    _append(tags, "ego_action", behavior.get("ego_action"))
    _append(tags, "other_actor_action", behavior.get("other_actor_action"))
    _append(tags, "avoidance_actor", behavior.get("avoidance_actor"))
    _append(tags, "collision_target", collision.get("target"))
    _append(tags, "collision_geometry", collision.get("geometry"))
    _append(tags, "collision_severity", collision.get("severity"))
    _append(tags, "near_miss_type", near_miss.get("type"))
    _append(tags, "review_required", review.get("required"))
    _append(tags, "review_confidence", review.get("confidence"))
    _append(tags, "briefing_timeline", briefing.get("timeline"))
    _append(tags, "briefing_actor", briefing.get("actor"))
    _append_many(tags, "evidence", briefing.get("evidence"))
    _append(tags, "model_version", runtime.get("model_version"))
    _append(tags, "processing_seconds", runtime.get("processing_seconds"))
    _append(tags, "model", outcome.get("model"))
    return tags


def _label_confidence(outcome: dict[str, Any], label: str) -> float | None:
    score_key = f"{label}_score"
    if isinstance(outcome.get(score_key), int | float):
        return float(outcome[score_key])

    scores = [
        float(value)
        for key, value in outcome.items()
        if key.endswith("_score") and isinstance(value, int | float)
    ]
    return max(scores) if scores else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _append(tags: list[tuple[str, str]], tag_type: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        tags.append((tag_type, text))


def _append_many(tags: list[tuple[str, str]], tag_type: str, values: Any) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        _append(tags, tag_type, value)
