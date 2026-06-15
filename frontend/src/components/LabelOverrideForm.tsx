import { useState } from 'react'
import type { FormEvent } from 'react'
import { Link } from 'react-router-dom'
import type { ClassificationLabel, IncidentLabel } from '../types/incident'
import { overrideLabel } from '../api/incidents'
import { useAuth } from '../auth/AuthContext'
import LabelBadge from './LabelBadge'

const OPTIONS: { value: ClassificationLabel; label: string }[] = [
  { value: 'safe', label: 'Safe' },
  { value: 'near_miss', label: 'Near miss' },
  { value: 'collision', label: 'Collision' },
]

interface LabelOverrideFormProps {
  incidentId: string
  currentLabel: IncidentLabel | null
  onUpdated: () => void
}

export default function LabelOverrideForm({
  incidentId,
  currentLabel,
  onUpdated,
}: LabelOverrideFormProps) {
  const { user } = useAuth()
  const [value, setValue] = useState<ClassificationLabel>(
    currentLabel?.value ?? 'safe',
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!user) {
      setError('Log in to save a label override.')
      return
    }
    setSaving(true)
    setError(null)
    setSuccess(false)
    try {
      await overrideLabel(incidentId, {
        value,
      })
      setSuccess(true)
      onUpdated()
    } catch {
      setError('Failed to save label override.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card annotation-card">
      <h2>Annotate classification</h2>
      <p className="text-muted">
        Current: <LabelBadge value={currentLabel?.value} source={currentLabel?.source} />
      </p>
      {!user && (
        <p className="text-muted">
          <Link to="/login">Log in</Link> to save classification edits.
        </p>
      )}
      <form onSubmit={handleSubmit} className="annotation-form">
        <div className="form-row">
          <label htmlFor="label-override">Override label</label>
          <select
            id="label-override"
            value={value}
            onChange={(e) => setValue(e.target.value as ClassificationLabel)}
          >
            {OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        {error && <p className="form-error">{error}</p>}
        {success && <p className="form-success">Label updated.</p>}
        <button type="submit" className="btn btn-primary" disabled={saving || !user}>
          {saving ? 'Saving…' : 'Save override'}
        </button>
      </form>
    </section>
  )
}
