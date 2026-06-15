import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { overrideSummary } from '../api/incidents'
import { useAuth } from '../auth/AuthContext'

interface SummaryOverrideFormProps {
  incidentId: string
  currentSummary: string | null | undefined
  onUpdated: () => void
}

export default function SummaryOverrideForm({
  incidentId,
  currentSummary,
  onUpdated,
}: SummaryOverrideFormProps) {
  const { user } = useAuth()
  const [text, setText] = useState(currentSummary ?? '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  useEffect(() => {
    setText(currentSummary ?? '')
  }, [currentSummary])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!user) {
      setError('Log in to save summary edits.')
      return
    }
    if (!text.trim()) {
      setError('Summary cannot be empty.')
      return
    }

    setSaving(true)
    setError(null)
    setSuccess(false)
    try {
      await overrideSummary(incidentId, { text: text.trim() })
      setSuccess(true)
      onUpdated()
    } catch {
      setError('Failed to save summary edit.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card annotation-card">
      <h2>Edit summary</h2>
      {!user && (
        <p className="text-muted">
          <Link to="/login">Log in</Link> to save summary edits.
        </p>
      )}
      <form onSubmit={handleSubmit} className="annotation-form">
        <div className="form-row">
          <label htmlFor="summary-override">Summary</label>
          <textarea
            id="summary-override"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>
        {error && <p className="form-error">{error}</p>}
        {success && <p className="form-success">Summary updated.</p>}
        <button type="submit" className="btn btn-primary" disabled={saving || !user}>
          {saving ? 'Saving...' : 'Save summary'}
        </button>
      </form>
    </section>
  )
}
