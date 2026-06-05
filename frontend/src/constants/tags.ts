export const SEMANTIC_TAG_TYPES = [
  { value: 'actor_type', label: 'Actor type' },
  { value: 'scenario_type', label: 'Scenario type' },
  { value: 'road_type', label: 'Road type' },
  { value: 'weather', label: 'Weather' },
  { value: 'avoidance_behaviour', label: 'Avoidance behaviour' },
  { value: 'collision_geometry', label: 'Collision geometry' },
  { value: 'near_miss_type', label: 'Near-miss type' },
  { value: 'context', label: 'Context' },
] as const

export const LABEL_COLORS: Record<string, string> = {
  safe: '#16a34a',
  near_miss: '#ca8a04',
  collision: '#dc2626',
  unclassified: '#6b7280',
}
