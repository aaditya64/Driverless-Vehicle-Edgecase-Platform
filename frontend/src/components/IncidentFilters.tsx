import { useEffect, useState } from 'react'
import type { ListIncidentsParams } from '../api/incidents'
import { getTagTypes, getTagValues, listIncidents } from '../api/incidents'
import type { TagTypeOption, TagValuesResponse } from '../api/incidents'
import type { IncidentSummary } from '../types/incident'

const FINITE_TAG_VALUE_LIMIT = 100

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

function labelizeTagType(value: string): string {
  return value.replace(/[_-]/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

function deriveTagMetadata(incidents: IncidentSummary[]) {
  const valuesByType = new Map<string, Set<string>>()

  incidents.forEach((incident) => {
    incident.tags?.forEach((tag) => {
      const tagType = tag.tag_type.trim()
      const tagValue = tag.tag_value.trim()
      if (!tagType || !tagValue) return
      const values = valuesByType.get(tagType) ?? new Set<string>()
      values.add(tagValue)
      valuesByType.set(tagType, values)
    })
  })

  const tagTypes = Array.from(valuesByType.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([value, values]) => ({
      value,
      label: labelizeTagType(value),
      value_count: values.size,
      has_value_options:
        values.size > 0 &&
        values.size <= FINITE_TAG_VALUE_LIMIT &&
        Array.from(values).every((tagValue) => tagValue.length <= 120),
    }))

  const tagValuesByType = Object.fromEntries(
    Array.from(valuesByType.entries()).map(([tagType, values]) => [
      tagType,
      Array.from(values).sort((a, b) => a.localeCompare(b)),
    ]),
  )

  return { tagTypes, tagValuesByType }
}

export default function IncidentFilters({
  filters,
  onChange,
  showLocationFilter = false,
  hasLocation,
  onHasLocationChange,
}: IncidentFiltersProps) {
  const [tagTypes, setTagTypes] = useState<TagTypeOption[]>([])
  const [derivedTagValues, setDerivedTagValues] = useState<Record<string, string[]>>({})
  const [tagValues, setTagValues] = useState<TagValuesResponse | null>(null)
  const [tagMetadataError, setTagMetadataError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    Promise.allSettled([getTagTypes(), listIncidents()])
      .then(([apiTypesResult, incidentsResult]) => {
        if (!active) return

        if (incidentsResult.status === 'fulfilled') {
          const derived = deriveTagMetadata(incidentsResult.value)
          setDerivedTagValues(derived.tagValuesByType)
          if (derived.tagTypes.length > 0) {
            setTagTypes(derived.tagTypes)
            setTagMetadataError(null)
          }
        }

        if (apiTypesResult.status === 'fulfilled' && apiTypesResult.value.length > 0) {
          setTagTypes(apiTypesResult.value)
          setTagMetadataError(null)
          return
        }

        if (incidentsResult.status !== 'fulfilled') {
          setTagMetadataError('Could not load tag metadata.')
        }
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    let active = true
    if (!filters.tagType) return

    getTagValues(filters.tagType)
      .then((nextValues) => {
        if (active) setTagValues(nextValues)
      })
      .catch(() => {
        if (active) setTagValues(null)
      })
    return () => {
      active = false
    }
  }, [filters.tagType])

  const set = (patch: Partial<IncidentFilterState>) => onChange({ ...filters, ...patch })
  const selectedTagType = tagTypes.find((t) => t.value === filters.tagType)
  const derivedValuesForSelectedType = filters.tagType
    ? derivedTagValues[filters.tagType] ?? []
    : []
  const apiValuesForSelectedType =
    tagValues?.tag_type === filters.tagType ? tagValues.values : []
  const tagValueOptions =
    tagValues?.tag_type === filters.tagType &&
    tagValues.has_value_options &&
    apiValuesForSelectedType.length > 0
      ? apiValuesForSelectedType
      : derivedValuesForSelectedType
  const useTagValueSelect =
    Boolean(filters.tagType) &&
    Boolean(selectedTagType?.has_value_options || derivedValuesForSelectedType.length > 0) &&
    tagValueOptions.length > 0 &&
    tagValueOptions.length <= FINITE_TAG_VALUE_LIMIT

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
          onChange={(e) => set({ tagType: e.target.value, tagValue: '' })}
        >
          <option value="">{tagTypes.length ? 'All' : 'Loading tag types...'}</option>
          {tagTypes.map((t) => (
            <option key={t.value} value={t.value}>
              {t.value}
            </option>
          ))}
        </select>
        {tagMetadataError && <span className="filter-hint-error">{tagMetadataError}</span>}
      </div>
      <div className="filter-group">
        <label htmlFor="filter-tag-value">Tag value</label>
        {useTagValueSelect ? (
          <select
            id="filter-tag-value"
            value={filters.tagValue}
            onChange={(e) => set({ tagValue: e.target.value })}
          >
            <option value="">All</option>
            {tagValueOptions.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        ) : (
          <input
            id="filter-tag-value"
            type="text"
            placeholder={filters.tagType ? 'Search tag value' : 'Select a tag type first'}
            value={filters.tagValue}
            onChange={(e) => set({ tagValue: e.target.value })}
          />
        )}
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
