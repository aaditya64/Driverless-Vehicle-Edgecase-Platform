import { useState } from 'react'

interface ContextTagInputProps {
  tags: string[]
  onChange: (tags: string[]) => void
}

export default function ContextTagInput({ tags, onChange }: ContextTagInputProps) {
  const [input, setInput] = useState('')

  const addTag = () => {
    const value = input.trim()
    if (!value || tags.includes(value)) return
    onChange([...tags, value])
    setInput('')
  }

  const removeTag = (tag: string) => {
    onChange(tags.filter((t) => t !== tag))
  }

  return (
    <div className="context-tags">
      <label htmlFor="context-tag-input">Context tags</label>
      <p className="text-muted form-hint">
        Optional labels for search and filtering (e.g. urban, motorway, rain).
      </p>
      <div className="context-tags-input-row">
        <input
          id="context-tag-input"
          type="text"
          placeholder="Add a tag and press Enter"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              addTag()
            }
          }}
        />
        <button type="button" className="btn btn-ghost btn-sm" onClick={addTag}>
          Add
        </button>
      </div>
      {tags.length > 0 && (
        <ul className="context-tags-list">
          {tags.map((tag) => (
            <li key={tag}>
              {tag}
              <button type="button" aria-label={`Remove ${tag}`} onClick={() => removeTag(tag)}>
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
