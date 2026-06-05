import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { createIncident } from '../api/incidents'
import VideoDropzone from '../components/VideoDropzone'
import LocationPicker from '../components/LocationPicker'
import ContextTagInput from '../components/ContextTagInput'

export default function UploadIncident() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [narrative, setNarrative] = useState('')
  const [lat, setLat] = useState<number | null>(null)
  const [lng, setLng] = useState<number | null>(null)
  const [contextTags, setContextTags] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      setError('Select a video file to upload.')
      return
    }

    const formData = new FormData()
    formData.append('video_file', file)
    if (narrative.trim()) formData.append('narrative', narrative.trim())
    if (lat != null) formData.append('location_lat', String(lat))
    if (lng != null) formData.append('location_lng', String(lng))
    if (contextTags.length > 0) {
      formData.append('context_tags', JSON.stringify(contextTags))
    }

    setSubmitting(true)
    setError(null)
    try {
      const created = await createIncident(formData)
      navigate(`/incidents/${created.id}`)
    } catch {
      setError('Upload failed. Check that the API and MinIO are running.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page page-narrow">
      <div className="page-header">
        <div>
          <h1>Upload incident</h1>
          <p className="text-muted">
            Drag-and-drop a dashcam video with optional location pin, narrative, and context
            tags.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="upload-form">
        <VideoDropzone file={file} onFileChange={setFile} />

        <div className="card form-section">
          <h2>Context</h2>
          <div className="form-row">
            <label htmlFor="narrative">Narrative description</label>
            <textarea
              id="narrative"
              rows={4}
              placeholder="What happened? Road conditions, actors, etc."
              value={narrative}
              onChange={(e) => setNarrative(e.target.value)}
            />
          </div>

          <ContextTagInput tags={contextTags} onChange={setContextTags} />
        </div>

        <div className="card form-section">
          <LocationPicker
            lat={lat}
            lng={lng}
            onChange={(newLat, newLng) => {
              setLat(newLat)
              setLng(newLng)
            }}
          />
          <div className="form-row form-row-split">
            <div>
              <label htmlFor="lat">Latitude</label>
              <input
                id="lat"
                type="number"
                step="any"
                placeholder="51.5074"
                value={lat ?? ''}
                onChange={(e) => setLat(e.target.value ? parseFloat(e.target.value) : null)}
              />
            </div>
            <div>
              <label htmlFor="lng">Longitude</label>
              <input
                id="lng"
                type="number"
                step="any"
                placeholder="-0.1278"
                value={lng ?? ''}
                onChange={(e) => setLng(e.target.value ? parseFloat(e.target.value) : null)}
              />
            </div>
          </div>
        </div>

        {error && <p className="form-error">{error}</p>}

        <div className="form-actions">
          <Link to="/" className="btn btn-ghost">
            Cancel
          </Link>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? 'Uploading…' : 'Submit for analysis'}
          </button>
        </div>
      </form>
    </div>
  )
}
