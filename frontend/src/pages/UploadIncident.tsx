import { useState } from 'react'
import api from '../api'

export default function UploadIncident() {
  const [narrative, setNarrative] = useState('')
  const [lat, setLat] = useState('')
  const [lng, setLng] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async () => {
    await api.post('/incidents', {
      narrative,
      location_lat: lat ? parseFloat(lat) : null,
      location_lng: lng ? parseFloat(lng) : null
    })
    setSubmitted(true)
  }

  if (submitted) return <p>Incident submitted successfully.</p>

  return (
    <div>
      <h1>Upload Incident</h1>
      <div>
        <label>Narrative</label><br />
        <textarea value={narrative} onChange={e => setNarrative(e.target.value)} rows={4} cols={50} />
      </div>
      <div>
        <label>Latitude</label><br />
        <input value={lat} onChange={e => setLat(e.target.value)} />
      </div>
      <div>
        <label>Longitude</label><br />
        <input value={lng} onChange={e => setLng(e.target.value)} />
      </div>
      <button onClick={handleSubmit}>Submit</button>
    </div>
  )
}