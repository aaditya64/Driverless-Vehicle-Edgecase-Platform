import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function Layout() {
  const { user, logout } = useAuth()

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <NavLink to="/" className="app-brand">
            Edge-Case Intelligence
          </NavLink>
          <nav className="app-nav">
            <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>
              Incidents
            </NavLink>
            <NavLink to="/map" className={({ isActive }) => (isActive ? 'active' : '')}>
              Map
            </NavLink>
            <NavLink to="/upload" className={({ isActive }) => (isActive ? 'active' : '')}>
              Upload
            </NavLink>
            {user ? (
              <button type="button" className="nav-button" onClick={logout}>
                {user.display_name} · Logout
              </button>
            ) : (
              <NavLink to="/login" className={({ isActive }) => (isActive ? 'active' : '')}>
                Login
              </NavLink>
            )}
          </nav>
        </div>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  )
}
