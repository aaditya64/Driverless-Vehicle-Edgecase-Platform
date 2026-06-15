import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import type { AuthUser } from '../api/auth'
import { getMe, login as loginRequest, signup as signupRequest } from '../api/auth'

interface AuthContextValue {
  user: AuthUser | null
  token: string | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string, displayName?: string) => Promise<void>
  logout: () => void
}

const TOKEN_KEY = 'edgecase_auth_token'
const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    if (!token) {
      setUser(null)
      setLoading(false)
      return
    }

    setLoading(true)
    getMe()
      .then((nextUser) => {
        if (active) setUser(nextUser)
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY)
        if (active) {
          setToken(null)
          setUser(null)
        }
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [token])

  const persistAuth = (nextToken: string, nextUser: AuthUser) => {
    localStorage.setItem(TOKEN_KEY, nextToken)
    setToken(nextToken)
    setUser(nextUser)
  }

  const login = useCallback(async (email: string, password: string) => {
    const response = await loginRequest({ email, password })
    persistAuth(response.token, response.user)
  }, [])

  const signup = useCallback(
    async (email: string, password: string, displayName?: string) => {
      const response = await signupRequest({
        email,
        password,
        display_name: displayName,
      })
      persistAuth(response.token, response.user)
    },
    [],
  )

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setUser(null)
  }, [])

  const value = useMemo(
    () => ({ user, token, loading, login, signup, logout }),
    [user, token, loading, login, signup, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider')
  }
  return context
}
