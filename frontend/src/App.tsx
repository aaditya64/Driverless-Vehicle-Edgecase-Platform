import { useState } from 'react'
import IncidentList from './pages/IncidentList'
import UploadIncident from './pages/UploadIncident'

export default function App() {
  const [page, setPage] = useState<'list' | 'upload'>('list')

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: '800px', margin: '0 auto', padding: '24px' }}>
      <nav style={{ marginBottom: '24px' }}>
        <button onClick={() => setPage('list')} style={{ marginRight: '12px' }}>Incident List</button>
        <button onClick={() => setPage('upload')}>Upload Incident</button>
      </nav>
      {page === 'list' && <IncidentList />}
      {page === 'upload' && <UploadIncident />}
    </div>
  )
}