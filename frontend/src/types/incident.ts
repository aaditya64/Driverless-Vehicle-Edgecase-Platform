export type IncidentStatus = 'waiting' | 'processing' | 'completed' | 'failed'
export type ClassificationLabel = 'safe' | 'near_miss' | 'collision'

export interface IncidentLabel {
  value: ClassificationLabel
  source: 'model' | 'human'
  confidence: number | null
}

export interface IncidentTag {
  tag_type: string
  tag_value: string
}

export interface IncidentSummary {
  id: string
  status: IncidentStatus
  narrative: string | null
  location_lat: number | null
  location_lng: number | null
  uploaded_at: string
  label: IncidentLabel | null
  tags: IncidentTag[]
}

export interface IncidentDetail extends IncidentSummary {
  summary?: string | null
  risk_timeline?: number[] | null
  video_url?: string
}

export interface LabelOverridePayload {
  value: ClassificationLabel
  changed_by: string
}

export interface TagOverridePayload {
  tags: IncidentTag[]
  changed_by: string
}
