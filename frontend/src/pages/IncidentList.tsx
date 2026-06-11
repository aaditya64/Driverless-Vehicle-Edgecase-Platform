import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { deleteIncident, listIncidents } from '../api/incidents'
import type { IncidentSummary } from '../types/incident'
import StatusBadge from '../components/StatusBadge'
import LabelBadge from '../components/LabelBadge'
import IncidentFilters, {
  DEFAULT_FILTERS,
  filtersToParams,
  type IncidentFilterState,
} from '../components/IncidentFilters'
import { usePolling } from '../hooks/usePolling'

export default function IncidentList() {
  const [incidents, setIncidents] = useState<IncidentSummary[]>([])
  const [filters, setFilters] = useState<IncidentFilterState>(DEFAULT_FILTERS)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    setError(null)
    try {
      const data = await listIncidents(filtersToParams(filters))
      setIncidents(data)
    } catch {
      setError('Could not load incidents. Is the API running?')
    } finally {
      if (!silent) setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    load()
  }, [load])

  const hasActiveProcessing = incidents.some(
    (i) => i.status === 'waiting' || i.status === 'processing',
  )
  usePolling(() => load(true), 5000, hasActiveProcessing)

  const handleDelete = async (incident: IncidentSummary) => {
    const confirmed = window.confirm(
      'Remove this incident record and delete its uploaded video?',
    )
    if (!confirmed) return

    setDeletingId(incident.id)
    setError(null)
    try {
      await deleteIncident(incident.id)
      setIncidents((current) => current.filter((i) => i.id !== incident.id))
    } catch {
      setError('Could not remove incident. Check the API logs and S3 permissions.')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Incidents</h1>
          <p className="text-muted">
            Queryable incident database — search, filter, and sort records.
            {hasActiveProcessing && (
              <span className="polling-indicator"> · Auto-refreshing status</span>
            )}
          </p>
        </div>
        <Link to="/upload" className="btn btn-primary">
          Upload video
        </Link>
      </div>

      <IncidentFilters filters={filters} onChange={setFilters} />

      {loading && <p className="text-muted">Loading…</p>}
      {error && <p className="form-error">{error}</p>}
      {!loading && !error && incidents.length === 0 && (
        <div className="empty-state card">
          <p>No incidents match your filters.</p>
          <Link to="/upload">Upload your first video</Link>
        </div>
      )}

      {!loading && !error && incidents.length > 0 && (
        <div className="incident-table-wrap card">
          <table className="incident-table">
            <thead>
              <tr>
                <th>Uploaded</th>
                <th>Status</th>
                <th>Classification</th>
                <th>Tags</th>
                <th>Narrative</th>
                <th>Location</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {incidents.map((i) => (
                <tr key={i.id}>
                  <td>{new Date(i.uploaded_at).toLocaleString()}</td>
                  <td>
                    <StatusBadge status={i.status} />
                  </td>
                  <td>
                    <LabelBadge
                      value={i.label?.value}
                      source={i.label?.source}
                    />
                  </td>
                  <td className="tags-cell">
                    {i.tags?.length ? (
                      <span className="tags-cell-preview">
                        {i.tags
                          .slice(0, 2)
                          .map((t) => t.tag_value)
                          .join(', ')}
                        {i.tags.length > 2 ? ` +${i.tags.length - 2}` : ''}
                      </span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td className="narrative-cell">
                    {i.narrative ? (
                      i.narrative.length > 80 ? `${i.narrative.slice(0, 80)}…` : i.narrative
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    {i.location_lat != null && i.location_lng != null ? (
                      `${i.location_lat.toFixed(4)}, ${i.location_lng.toFixed(4)}`
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    <div className="table-actions">
                      <Link to={`/incidents/${i.id}`} className="btn btn-ghost btn-sm">
                        View
                      </Link>
                      <button
                        type="button"
                        className="btn btn-danger btn-sm"
                        disabled={deletingId === i.id}
                        onClick={() => handleDelete(i)}
                      >
                        {deletingId === i.id ? 'Removing…' : 'Remove'}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
