import type { ClassificationLabel } from '../types/incident'

const LABEL_TEXT: Record<ClassificationLabel, string> = {
  safe: 'Safe',
  near_miss: 'Near miss',
  collision: 'Collision',
}

interface LabelBadgeProps {
  value: ClassificationLabel | string | null | undefined
  source?: string | null
}

export default function LabelBadge({ value, source }: LabelBadgeProps) {
  if (!value) return <span className="badge badge-muted">Unclassified</span>
  const key = value as ClassificationLabel
  const text = LABEL_TEXT[key] ?? value
  return (
    <span className={`badge badge-label badge-label-${value}`}>
      {text}
      {source && <span className="badge-source"> · {source}</span>}
    </span>
  )
}
