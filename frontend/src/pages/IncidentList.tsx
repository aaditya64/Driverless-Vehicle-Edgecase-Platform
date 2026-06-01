import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listIncidents } from '../api/incidents'
import type { IncidentSummary } from '../types/incident'
import StatusBadge from '../components/StatusBadge'
import LabelBadge from '../components/LabelBadge'

export default function IncidentList() {
  const [incidents, setIncidents] = useState<IncidentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState('')
  const [labelFilter, setLabelFilter] = useState('')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await listIncidents({
        status: statusFilter || undefined,
        label: labelFilter || undefined,
        order,
      })
      setIncidents(data)
    } catch {
      setError('Could not load incidents. Is the API running?')
    } finally {
      setLoading(false)
    }
  }, [statusFilter, labelFilter, order])

  useEffect(() => {
    load()
  }, [load])

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Incidents</h1>
          <p className="text-muted">Search and review analysed dashcam events.</p>
        </div>
        <Link to="/upload" className="btn btn-primary">
          Upload video
        </Link>
      </div>

      <div className="filters card">
        <div className="filter-group">
          <label htmlFor="filter-status">Status</label>
          <select
            id="filter-status"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All</option>
            <option value="waiting">Waiting</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        <div className="filter-group">
          <label htmlFor="filter-label">Classification</label>
          <select
            id="filter-label"
            value={labelFilter}
            onChange={(e) => setLabelFilter(e.target.value)}
          >
            <option value="">All</option>
            <option value="safe">Safe</option>
            <option value="near_miss">Near miss</option>
            <option value="collision">Collision</option>
          </select>
        </div>
        <div className="filter-group">
          <label htmlFor="filter-order">Sort by date</label>
          <select
            id="filter-order"
            value={order}
            onChange={(e) => setOrder(e.target.value as 'asc' | 'desc')}
          >
            <option value="desc">Newest first</option>
            <option value="asc">Oldest first</option>
          </select>
        </div>
      </div>

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
                    <Link to={`/incidents/${i.id}`} className="btn btn-ghost btn-sm">
                      View
                    </Link>
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
