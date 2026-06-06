import { useEffect, useState } from 'react'
import type { IncidentTag } from '../types/incident'
import { overrideTags } from '../api/incidents'
import { SEMANTIC_TAG_TYPES } from '../constants/tags'

interface TagRow {
  tag_type: string
  tag_value: string
}

interface TagOverrideFormProps {
  incidentId: string
  currentTags: IncidentTag[]
  onUpdated: () => void
}

function tagsToRows(tags: IncidentTag[]): TagRow[] {
  if (tags.length === 0) return [{ tag_type: 'context', tag_value: '' }]
  return tags.map((t) => ({ tag_type: t.tag_type, tag_value: t.tag_value }))
}

export default function TagOverrideForm({
  incidentId,
  currentTags,
  onUpdated,
}: TagOverrideFormProps) {
  const [rows, setRows] = useState<TagRow[]>(() => tagsToRows(currentTags))
  const [reviewer, setReviewer] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  useEffect(() => {
    setRows(tagsToRows(currentTags))
  }, [currentTags])

  const updateRow = (index: number, patch: Partial<TagRow>) => {
    setRows((prev) => prev.map((r, i) => (i === index ? { ...r, ...patch } : r)))
  }

  const addRow = () => setRows((prev) => [...prev, { tag_type: 'context', tag_value: '' }])

  const removeRow = (index: number) => {
    setRows((prev) => (prev.length === 1 ? prev : prev.filter((_, i) => i !== index)))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!reviewer.trim()) {
      setError('Enter your name as reviewer.')
      return
    }
    const tags = rows.filter((r) => r.tag_type.trim() && r.tag_value.trim())
    setSaving(true)
    setError(null)
    setSuccess(false)
    try {
      await overrideTags(incidentId, {
        tags,
        changed_by: reviewer.trim(),
      })
      setSuccess(true)
      onUpdated()
    } catch {
      setError('Failed to save tag overrides.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card annotation-card">
      <h2>Annotate tags</h2>
      <p className="text-muted">
        Override model-generated semantic tags or add context tags. Changes are saved to the
        database.
      </p>
      <form onSubmit={handleSubmit} className="annotation-form">
        <div className="tag-override-rows">
          {rows.map((row, index) => (
            <div key={index} className="tag-override-row">
              <select
                value={row.tag_type}
                onChange={(e) => updateRow(index, { tag_type: e.target.value })}
                aria-label={`Tag type ${index + 1}`}
              >
                {SEMANTIC_TAG_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
              <input
                type="text"
                placeholder="Tag value"
                value={row.tag_value}
                onChange={(e) => updateRow(index, { tag_value: e.target.value })}
              />
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => removeRow(index)}
                disabled={rows.length === 1}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
        <button type="button" className="btn btn-ghost btn-sm" onClick={addRow}>
          Add tag
        </button>
        <div className="form-row">
          <label htmlFor="tag-reviewer">Reviewer</label>
          <input
            id="tag-reviewer"
            type="text"
            placeholder="Your name"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
          />
        </div>
        {error && <p className="form-error">{error}</p>}
        {success && <p className="form-success">Tags updated.</p>}
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? 'Saving…' : 'Save tag overrides'}
        </button>
      </form>
    </section>
  )
}
