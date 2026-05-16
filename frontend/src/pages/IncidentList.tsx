import { useEffect, useState } from 'react'
import api from '../api'

interface Incident {
  id: string
  narrative: string | null
  status: string
  location_lat: number | null
  location_lng: number | null
  uploaded_at: string
}

export default function IncidentList() {
  const [incidents, setIncidents] = useState<Incident[]>([])

  useEffect(() => {
    api.get('/incidents').then(res => setIncidents(res.data.incidents))
  }, [])

  return (
    <div>
      <h1>Incidents</h1>
      {incidents.length === 0 && <p>No incidents yet.</p>}
      {incidents.map(i => (
        <div key={i.id} style={{ border: '1px solid #ccc', margin: '8px', padding: '12px' }}>
          <p><strong>ID:</strong> {i.id}</p>
          <p><strong>Status:</strong> {i.status}</p>
          <p><strong>Narrative:</strong> {i.narrative ?? '—'}</p>
          <p><strong>Uploaded:</strong> {new Date(i.uploaded_at).toLocaleString()}</p>
        </div>
      ))}
    </div>
  )
}