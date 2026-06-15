import api from '../api'

export interface AuthUser {
  id: string
  email: string
  display_name: string
  created_at?: string
}

export interface AuthResponse {
  token: string
  user: AuthUser
}

export async function signup(payload: {
  email: string
  password: string
  display_name?: string
}): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/signup', payload)
  return data
}

export async function login(payload: {
  email: string
  password: string
}): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/login', payload)
  return data
}

export async function getMe(): Promise<AuthUser> {
  const { data } = await api.get<{ user: AuthUser }>('/auth/me')
  return data.user
}
