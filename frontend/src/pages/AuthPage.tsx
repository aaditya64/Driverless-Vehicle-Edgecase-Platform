import { useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function AuthPage() {
  const { login, signup } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [mode, setMode] = useState<'login' | 'signup'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const from = (location.state as { from?: string } | null)?.from ?? '/'

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      if (mode === 'login') {
        await login(email.trim(), password)
      } else {
        await signup(email.trim(), password, displayName.trim() || undefined)
      }
      navigate(from, { replace: true })
    } catch {
      setError(mode === 'login' ? 'Login failed.' : 'Sign up failed.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page page-narrow">
      <div className="page-header">
        <div>
          <h1>{mode === 'login' ? 'Log in' : 'Sign up'}</h1>
          <p className="text-muted">
            Use an account to upload videos and track edits to ML outputs.
          </p>
        </div>
      </div>

      <form className="card auth-form" onSubmit={handleSubmit}>
        {mode === 'signup' && (
          <div className="form-row">
            <label htmlFor="display-name">Display name</label>
            <input
              id="display-name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Aaditya"
            />
          </div>
        )}
        <div className="form-row">
          <label htmlFor="auth-email">Email</label>
          <input
            id="auth-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="form-row">
          <label htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={8}
            required
          />
        </div>
        {error && <p className="form-error">{error}</p>}
        <button className="btn btn-primary" type="submit" disabled={submitting}>
          {submitting ? 'Working...' : mode === 'login' ? 'Log in' : 'Create account'}
        </button>
      </form>

      <p className="text-muted">
        {mode === 'login' ? 'No account yet?' : 'Already have an account?'}{' '}
        <button
          className="link-button"
          type="button"
          onClick={() => {
            setMode(mode === 'login' ? 'signup' : 'login')
            setError(null)
          }}
        >
          {mode === 'login' ? 'Sign up' : 'Log in'}
        </button>
      </p>
      <Link to="/" className="back-link">
        Back to incidents
      </Link>
    </div>
  )
}
