import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { listIncidents } from '../api/incidents'
import type { IncidentSummary } from '../types/incident'
import IncidentFilters, {
  DEFAULT_FILTERS,
  filtersToParams,
  type IncidentFilterState,
} from '../components/IncidentFilters'
import { LABEL_COLORS } from '../constants/tags'
import { usePolling } from '../hooks/usePolling'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN as string | undefined
const DEFAULT_CENTER: [number, number] = [-0.1278, 51.5074]

function pinColor(incident: IncidentSummary): string {
  return LABEL_COLORS[incident.label?.value ?? 'unclassified'] ?? LABEL_COLORS.unclassified
}

export default function IncidentMap() {
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const markersRef = useRef<mapboxgl.Marker[]>([])
  const [incidents, setIncidents] = useState<IncidentSummary[]>([])
  const [filters, setFilters] = useState<IncidentFilterState>(DEFAULT_FILTERS)
  const [hasLocation, setHasLocation] = useState<boolean | undefined>(true)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true)
    setError(null)
    try {
      const data = await listIncidents({
        ...filtersToParams(filters),
        has_location: hasLocation,
      })
      setIncidents(data)
    } catch {
      setError('Could not load incidents for the map.')
    } finally {
      if (!silent) setLoading(false)
    }
  }, [filters, hasLocation])

  useEffect(() => {
    load()
  }, [load])

  const hasActiveProcessing = incidents.some(
    (i) => i.status === 'waiting' || i.status === 'processing',
  )
  usePolling(() => load(true), 5000, hasActiveProcessing)

  useEffect(() => {
    if (!MAPBOX_TOKEN || !mapContainerRef.current) return

    mapboxgl.accessToken = MAPBOX_TOKEN
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/streets-v12',
      center: DEFAULT_CENTER,
      zoom: 10,
    })
    map.addControl(new mapboxgl.NavigationControl(), 'top-right')
    mapRef.current = map

    return () => {
      markersRef.current.forEach((m) => m.remove())
      markersRef.current = []
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    markersRef.current.forEach((m) => m.remove())
    markersRef.current = []

    const located = incidents.filter(
      (i) => i.location_lat != null && i.location_lng != null,
    )

    located.forEach((incident) => {
      const el = document.createElement('div')
      el.className = 'map-pin'
      el.style.backgroundColor = pinColor(incident)

      const popup = new mapboxgl.Popup({ offset: 16 }).setHTML(`
        <div class="map-popup">
          <strong>${incident.label?.value?.replace('_', ' ') ?? 'Unclassified'}</strong>
          <span class="map-popup-status">${incident.status}</span>
          <p>${incident.narrative ? incident.narrative.slice(0, 100) : 'No narrative'}</p>
          <a href="/incidents/${incident.id}">View incident →</a>
        </div>
      `)

      const marker = new mapboxgl.Marker({ element: el })
        .setLngLat([incident.location_lng!, incident.location_lat!])
        .setPopup(popup)
        .addTo(map)

      markersRef.current.push(marker)
    })

    if (located.length > 0) {
      const bounds = new mapboxgl.LngLatBounds()
      located.forEach((i) => bounds.extend([i.location_lng!, i.location_lat!]))
      map.fitBounds(bounds, { padding: 60, maxZoom: 14 })
    }
  }, [incidents])

  if (!MAPBOX_TOKEN) {
    return (
      <div className="page">
        <h1>Incident map</h1>
        <div className="map-fallback card">
          <p>
            Add your Mapbox token to <code>frontend/.env</code>:
          </p>
          <pre>VITE_MAPBOX_TOKEN=pk.your_token_here</pre>
          <p>
            Get a free token at{' '}
            <a href="https://account.mapbox.com/" target="_blank" rel="noreferrer">
              mapbox.com
            </a>
            .
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Incident map</h1>
          <p className="text-muted">
            Classification-coloured pins. Filter by status, tags, dates, and location.
          </p>
        </div>
        <div className="map-legend">
          {Object.entries(LABEL_COLORS).map(([key, color]) => (
            <span key={key} className="map-legend-item">
              <span className="map-legend-swatch" style={{ background: color }} />
              {key.replace('_', ' ')}
            </span>
          ))}
        </div>
      </div>

      <IncidentFilters
        filters={filters}
        onChange={setFilters}
        showLocationFilter
        hasLocation={hasLocation}
        onHasLocationChange={setHasLocation}
      />

      {loading && <p className="text-muted">Loading map…</p>}
      {error && <p className="form-error">{error}</p>}

      <div ref={mapContainerRef} className="map-container map-container-full" />

      {!loading && incidents.length === 0 && (
        <p className="text-muted map-empty-hint">
          No incidents with a location match these filters.{' '}
          <Link to="/upload">Upload one with a map pin</Link>.
        </p>
      )}
    </div>
  )
}
