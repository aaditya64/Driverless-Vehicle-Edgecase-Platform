import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import RequireAuth from './components/RequireAuth'
import IncidentList from './pages/IncidentList'
import UploadIncident from './pages/UploadIncident'
import IncidentDetail from './pages/IncidentDetail'
import IncidentMap from './pages/IncidentMap'
import AuthPage from './pages/AuthPage'
import { AuthProvider } from './auth/AuthContext'
import './styles/app.css'

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<IncidentList />} />
            <Route path="map" element={<IncidentMap />} />
            <Route
              path="upload"
              element={
                <RequireAuth>
                  <UploadIncident />
                </RequireAuth>
              }
            />
            <Route path="incidents/:id" element={<IncidentDetail />} />
            <Route path="login" element={<AuthPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
