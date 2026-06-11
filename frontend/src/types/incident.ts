export type IncidentStatus = 'waiting' | 'processing' | 'completed' | 'failed'
export type ClassificationLabel = 'safe' | 'near_miss' | 'collision'

export interface IncidentLabel {
  value: ClassificationLabel
  source: string
  confidence: number | null
}

export interface IncidentTag {
  tag_type: string
  tag_value: string
}

export interface RiskTimelinePoint {
  frame_idx: number
  time_sec: number
  risk_score: number
  shake_energy?: number
  jerk_energy?: number
  valid_motion?: boolean
}

export interface RiskTimelinePayload {
  source?: string
  score_type?: string
  temporal_resolution?: string
  frame_count?: number
  fps?: number
  duration_sec?: number
  peak?: {
    frame_idx?: number
    time_sec?: number
    risk_score?: number
  }
  points: RiskTimelinePoint[]
}

export type RiskTimeline = number[] | RiskTimelinePayload

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
  risk_timeline?: RiskTimeline | null
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
