import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { createIncident } from '../api/incidents'
import VideoDropzone from '../components/VideoDropzone'

export default function UploadIncident() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [narrative, setNarrative] = useState('')
  const [lat, setLat] = useState('')
  const [lng, setLng] = useState('')
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
    if (lat) formData.append('location_lat', lat)
    if (lng) formData.append('location_lng', lng)

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
            Submit a dashcam video with optional location and narrative context.
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
          <div className="form-row form-row-split">
            <div>
              <label htmlFor="lat">Latitude</label>
              <input
                id="lat"
                type="number"
                step="any"
                placeholder="51.5074"
                value={lat}
                onChange={(e) => setLat(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="lng">Longitude</label>
              <input
                id="lng"
                type="number"
                step="any"
                placeholder="-0.1278"
                value={lng}
                onChange={(e) => setLng(e.target.value)}
              />
            </div>
          </div>
          <p className="text-muted form-hint">
            Map pin selection will be added in a later step; enter coordinates manually for now.
          </p>
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
