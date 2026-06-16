import api from '../api'
import type {
  IncidentDetail,
  IncidentSummary,
  LabelOverridePayload,
  SummaryOverridePayload,
  TagOverridePayload,
} from '../types/incident'

export interface TagTypeOption {
  value: string
  label: string
  value_count: number
  has_value_options: boolean
}

export interface TagValuesResponse {
  tag_type: string
  values: string[]
  value_count: number
  has_value_options: boolean
}

export interface ListIncidentsParams {
  label?: string
  status?: string
  q?: string
  date_from?: string
  date_to?: string
  tag_type?: string
  tag_value?: string
  has_location?: boolean
  order?: 'asc' | 'desc'
}

export async function listIncidents(
  params: ListIncidentsParams = {},
): Promise<IncidentSummary[]> {
  const { data } = await api.get<{ incidents: IncidentSummary[] }>('/incidents', {
    params,
  })
  return data.incidents
}

export async function exportIncidents(
  params: ListIncidentsParams = {},
): Promise<{ count: number; incidents: IncidentDetail[] }> {
  const { data } = await api.get<{ count: number; incidents: IncidentDetail[] }>(
    '/export',
    { params },
  )
  return data
}

export async function getTagTypes(): Promise<TagTypeOption[]> {
  const { data } = await api.get<{ tag_types: TagTypeOption[] }>('/tags/types')
  return data.tag_types
}

export async function getTagValues(tagType: string): Promise<TagValuesResponse> {
  const { data } = await api.get<TagValuesResponse>('/tags/values', {
    params: { tag_type: tagType },
  })
  return data
}

export async function getIncident(id: string): Promise<IncidentDetail> {
  const { data } = await api.get<IncidentDetail>(`/incidents/${id}`)
  return data
}

export async function createIncident(formData: FormData): Promise<IncidentSummary> {
  const { data } = await api.post<IncidentSummary>('/incidents', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function deleteIncident(incidentId: string): Promise<void> {
  await api.delete(`/incidents/${incidentId}`)
}

export async function overrideLabel(
  incidentId: string,
  payload: LabelOverridePayload,
): Promise<void> {
  await api.patch(`/incidents/${incidentId}/labels`, payload)
}

export async function overrideTags(
  incidentId: string,
  payload: TagOverridePayload,
): Promise<void> {
  await api.patch(`/incidents/${incidentId}/tags`, payload)
}

export async function overrideSummary(
  incidentId: string,
  payload: SummaryOverridePayload,
): Promise<void> {
  await api.patch(`/incidents/${incidentId}/summary`, payload)
}
