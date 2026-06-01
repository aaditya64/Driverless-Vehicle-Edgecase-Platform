import { useState } from 'react'
import type { ClassificationLabel, IncidentLabel } from '../types/incident'
import { overrideLabel } from '../api/incidents'
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
  const [value, setValue] = useState<ClassificationLabel>(
    currentLabel?.value ?? 'safe',
  )
  const [reviewer, setReviewer] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!reviewer.trim()) {
      setError('Enter your name as reviewer.')
      return
    }
    setSaving(true)
    setError(null)
    setSuccess(false)
    try {
      await overrideLabel(incidentId, {
        value,
        changed_by: reviewer.trim(),
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
        <div className="form-row">
          <label htmlFor="reviewer">Reviewer</label>
          <input
            id="reviewer"
            type="text"
            placeholder="Your name"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
          />
        </div>
        {error && <p className="form-error">{error}</p>}
        {success && <p className="form-success">Label updated.</p>}
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? 'Saving…' : 'Save override'}
        </button>
      </form>
    </section>
  )
}
