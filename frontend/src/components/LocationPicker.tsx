import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN as string | undefined
const DEFAULT_CENTER: [number, number] = [-0.1278, 51.5074]

interface LocationPickerProps {
  lat: number | null
  lng: number | null
  onChange: (lat: number, lng: number) => void
}

export default function LocationPicker({ lat, lng, onChange }: LocationPickerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const markerRef = useRef<mapboxgl.Marker | null>(null)

  useEffect(() => {
    if (!MAPBOX_TOKEN || !containerRef.current) return

    mapboxgl.accessToken = MAPBOX_TOKEN
    const center: [number, number] =
      lat != null && lng != null ? [lng, lat] : DEFAULT_CENTER

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: 'mapbox://styles/mapbox/streets-v12',
      center,
      zoom: lat != null && lng != null ? 12 : 10,
    })

    map.addControl(new mapboxgl.NavigationControl(), 'top-right')

    const marker = new mapboxgl.Marker({ draggable: true, color: '#7c3aed' })
    if (lat != null && lng != null) {
      marker.setLngLat([lng, lat]).addTo(map)
    }

    marker.on('dragend', () => {
      const pos = marker.getLngLat()
      onChange(pos.lat, pos.lng)
    })

    map.on('click', (e) => {
      marker.setLngLat(e.lngLat).addTo(map)
      onChange(e.lngLat.lat, e.lngLat.lng)
    })

    mapRef.current = map
    markerRef.current = marker

    return () => {
      marker.remove()
      map.remove()
      mapRef.current = null
      markerRef.current = null
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!markerRef.current || lat == null || lng == null) return
    markerRef.current.setLngLat([lng, lat])
    mapRef.current?.flyTo({ center: [lng, lat], zoom: 12 })
  }, [lat, lng])

  if (!MAPBOX_TOKEN) {
    return (
      <div className="map-fallback card">
        <p>
          Set <code>VITE_MAPBOX_TOKEN</code> in <code>frontend/.env</code> to enable the
          location pin map. Use the coordinate fields below in the meantime.
        </p>
      </div>
    )
  }

  return (
    <div className="location-picker">
      <label>Location pin</label>
      <p className="text-muted form-hint">Click the map or drag the pin to set incident location.</p>
      <div ref={containerRef} className="map-container map-container-picker" />
      {lat != null && lng != null && (
        <p className="text-muted map-coords">
          {lat.toFixed(5)}, {lng.toFixed(5)}
        </p>
      )}
    </div>
  )
}
