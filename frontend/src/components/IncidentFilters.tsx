import type { ListIncidentsParams } from '../api/incidents'
import { SEMANTIC_TAG_TYPES } from '../constants/tags'

export interface IncidentFilterState {
  q: string
  status: string
  label: string
  dateFrom: string
  dateTo: string
  tagType: string
  tagValue: string
  order: 'asc' | 'desc'
}

export const DEFAULT_FILTERS: IncidentFilterState = {
  q: '',
  status: '',
  label: '',
  dateFrom: '',
  dateTo: '',
  tagType: '',
  tagValue: '',
  order: 'desc',
}

export function filtersToParams(filters: IncidentFilterState): ListIncidentsParams {
  return {
    q: filters.q || undefined,
    status: filters.status || undefined,
    label: filters.label || undefined,
    date_from: filters.dateFrom ? `${filters.dateFrom}T00:00:00` : undefined,
    date_to: filters.dateTo ? `${filters.dateTo}T23:59:59` : undefined,
    tag_type: filters.tagType || undefined,
    tag_value: filters.tagValue || undefined,
    order: filters.order,
  }
}

interface IncidentFiltersProps {
  filters: IncidentFilterState
  onChange: (filters: IncidentFilterState) => void
  showLocationFilter?: boolean
  hasLocation?: boolean
  onHasLocationChange?: (value: boolean | undefined) => void
}

export default function IncidentFilters({
  filters,
  onChange,
  showLocationFilter = false,
  hasLocation,
  onHasLocationChange,
}: IncidentFiltersProps) {
  const set = (patch: Partial<IncidentFilterState>) => onChange({ ...filters, ...patch })

  return (
    <div className="filters card">
      <div className="filter-group filter-group-wide">
        <label htmlFor="filter-q">Search</label>
        <input
          id="filter-q"
          type="search"
          placeholder="Narrative or incident ID…"
          value={filters.q}
          onChange={(e) => set({ q: e.target.value })}
        />
      </div>
      <div className="filter-group">
        <label htmlFor="filter-status">Status</label>
        <select
          id="filter-status"
          value={filters.status}
          onChange={(e) => set({ status: e.target.value })}
        >
          <option value="">All</option>
          <option value="waiting">Waiting</option>
          <option value="processing">Processing</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="filter-label">Classification</label>
        <select
          id="filter-label"
          value={filters.label}
          onChange={(e) => set({ label: e.target.value })}
        >
          <option value="">All</option>
          <option value="safe">Safe</option>
          <option value="near_miss">Near miss</option>
          <option value="collision">Collision</option>
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="filter-date-from">From</label>
        <input
          id="filter-date-from"
          type="date"
          value={filters.dateFrom}
          onChange={(e) => set({ dateFrom: e.target.value })}
        />
      </div>
      <div className="filter-group">
        <label htmlFor="filter-date-to">To</label>
        <input
          id="filter-date-to"
          type="date"
          value={filters.dateTo}
          onChange={(e) => set({ dateTo: e.target.value })}
        />
      </div>
      <div className="filter-group">
        <label htmlFor="filter-tag-type">Tag type</label>
        <select
          id="filter-tag-type"
          value={filters.tagType}
          onChange={(e) => set({ tagType: e.target.value })}
        >
          <option value="">All</option>
          {SEMANTIC_TAG_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="filter-tag-value">Tag value</label>
        <input
          id="filter-tag-value"
          type="text"
          placeholder="e.g. pedestrian"
          value={filters.tagValue}
          onChange={(e) => set({ tagValue: e.target.value })}
        />
      </div>
      <div className="filter-group">
        <label htmlFor="filter-order">Sort</label>
        <select
          id="filter-order"
          value={filters.order}
          onChange={(e) => set({ order: e.target.value as 'asc' | 'desc' })}
        >
          <option value="desc">Newest first</option>
          <option value="asc">Oldest first</option>
        </select>
      </div>
      {showLocationFilter && onHasLocationChange && (
        <div className="filter-group">
          <label htmlFor="filter-location">On map</label>
          <select
            id="filter-location"
            value={hasLocation === undefined ? '' : hasLocation ? 'yes' : 'no'}
            onChange={(e) => {
              const v = e.target.value
              onHasLocationChange(v === '' ? undefined : v === 'yes')
            }}
          >
            <option value="">All</option>
            <option value="yes">Has location</option>
            <option value="no">No location</option>
          </select>
        </div>
      )}
    </div>
  )
}
