import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getIncident } from '../api/incidents'
import type { IncidentDetail as IncidentDetailType } from '../types/incident'
import StatusBadge from '../components/StatusBadge'
import LabelBadge from '../components/LabelBadge'
import RiskTimelineChart from '../components/RiskTimelineChart'
import LabelOverrideForm from '../components/LabelOverrideForm'
import TagOverrideForm from '../components/TagOverrideForm'
import { usePolling } from '../hooks/usePolling'

export default function IncidentDetail() {
  const { id } = useParams<{ id: string }>()
  const [incident, setIncident] = useState<IncidentDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (silent = false) => {
    if (!id) return
    if (!silent) setLoading(true)
    setError(null)
    try {
      setIncident(await getIncident(id))
    } catch {
      setError('Incident not found or API unavailable.')
    } finally {
      if (!silent) setLoading(false)
    }
  }, [id])

  useEffect(() => {
    load()
  }, [load])

  const isProcessing =
    incident?.status === 'waiting' || incident?.status === 'processing'
  usePolling(() => load(true), 5000, !!isProcessing)

  if (loading) {
    return (
      <div className="page">
        <p className="text-muted">Loading incident…</p>
      </div>
    )
  }

  if (error || !incident) {
    return (
      <div className="page">
        <p className="form-error">{error ?? 'Not found'}</p>
        <Link to="/">Back to incidents</Link>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <Link to="/" className="back-link">
            ← Incidents
          </Link>
          <h1>Incident detail</h1>
          <p className="text-muted mono-id">{incident.id}</p>
        </div>
        <div className="detail-status-wrap">
          <StatusBadge status={incident.status} />
          {isProcessing && (
            <span className="polling-indicator">Checking for updates…</span>
          )}
        </div>
      </div>

      <div className="detail-grid">
        <section className="card detail-video">
          <h2>Video</h2>
          {incident.video_url ? (
            <video
              key={incident.video_url}
              src={incident.video_url}
              controls
              className="incident-video"
            />
          ) : (
            <p className="text-muted">Video unavailable.</p>
          )}
        </section>

        <aside className="detail-sidebar">
          <section className="card">
            <h2>Metadata</h2>
            <dl className="meta-list">
              <dt>Uploaded</dt>
              <dd>{new Date(incident.uploaded_at).toLocaleString()}</dd>
              <dt>Classification</dt>
              <dd>
                <LabelBadge
                  value={incident.label?.value}
                  source={incident.label?.source}
                />
              </dd>
              <dt>Location</dt>
              <dd>
                {incident.location_lat != null && incident.location_lng != null
                  ? `${incident.location_lat}, ${incident.location_lng}`
                  : '—'}
              </dd>
              <dt>Narrative</dt>
              <dd>{incident.narrative ?? '—'}</dd>
            </dl>
          </section>

          <LabelOverrideForm
            incidentId={incident.id}
            currentLabel={incident.label}
            onUpdated={() => load(true)}
          />
        </aside>
      </div>

      {isProcessing && (
        <div className="card processing-banner">
          <p>
            <strong>Status: {incident.status}.</strong> ML outputs (risk timeline, tags,
            summary) will appear here when analysis completes. This page refreshes
            automatically.
          </p>
        </div>
      )}

      {incident.summary && (
        <section className="card">
          <h2>Summary</h2>
          <p className="summary-text">{incident.summary}</p>
        </section>
      )}

      {incident.tags && incident.tags.length > 0 && (
        <section className="card">
          <h2>Semantic tags</h2>
          <ul className="tag-list">
            {incident.tags.map((t) => (
              <li key={`${t.tag_type}-${t.tag_value}`}>
                <span className="tag-type">{t.tag_type}</span>
                <span className="tag-value">{t.tag_value}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <TagOverrideForm
        incidentId={incident.id}
        currentTags={incident.tags ?? []}
        onUpdated={() => load(true)}
      />

      {incident.risk_timeline && incident.risk_timeline.length > 0 && (
        <section className="card">
          <h2>Risk timeline</h2>
          <p className="text-muted">BADAS temporal risk signal across the analysed clip.</p>
          <RiskTimelineChart scores={incident.risk_timeline} />
        </section>
      )}
    </div>
  )
}
