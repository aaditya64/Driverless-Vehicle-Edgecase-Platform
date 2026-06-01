import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import IncidentList from './pages/IncidentList'
import UploadIncident from './pages/UploadIncident'
import IncidentDetail from './pages/IncidentDetail'
import './styles/app.css'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<IncidentList />} />
          <Route path="upload" element={<UploadIncident />} />
          <Route path="incidents/:id" element={<IncidentDetail />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
