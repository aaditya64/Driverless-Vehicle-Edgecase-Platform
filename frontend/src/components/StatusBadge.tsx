import type { IncidentStatus } from '../types/incident'

const STATUS_LABELS: Record<IncidentStatus, string> = {
  waiting: 'Waiting',
  processing: 'Processing',
  completed: 'Completed',
  failed: 'Failed',
}

interface StatusBadgeProps {
  status: string
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  const key = status as IncidentStatus
  const label = STATUS_LABELS[key] ?? status
  return <span className={`badge badge-status badge-status-${status}`}>{label}</span>
}
