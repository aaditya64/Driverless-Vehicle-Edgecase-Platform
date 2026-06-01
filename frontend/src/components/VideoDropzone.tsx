import { useCallback, useState } from 'react'

interface VideoDropzoneProps {
  file: File | null
  onFileChange: (file: File | null) => void
}

export default function VideoDropzone({ file, onFileChange }: VideoDropzoneProps) {
  const [dragOver, setDragOver] = useState(false)

  const acceptFile = useCallback(
    (candidate: File | undefined) => {
      if (!candidate) return
      if (!candidate.type.startsWith('video/') && !candidate.name.match(/\.(mp4|webm|mov|avi|mkv)$/i)) {
        return
      }
      onFileChange(candidate)
    },
    [onFileChange],
  )

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    acceptFile(e.dataTransfer.files[0])
  }

  return (
    <div
      className={`dropzone ${dragOver ? 'dropzone-active' : ''} ${file ? 'dropzone-has-file' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <input
        type="file"
        accept="video/*,.mp4,.webm,.mov,.avi,.mkv"
        className="dropzone-input"
        onChange={(e) => acceptFile(e.target.files?.[0])}
      />
      {file ? (
        <div className="dropzone-file">
          <strong>{file.name}</strong>
          <span>{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={(e) => {
              e.stopPropagation()
              onFileChange(null)
            }}
          >
            Remove
          </button>
        </div>
      ) : (
        <div className="dropzone-placeholder">
          <p>Drag and drop a dashcam video here</p>
          <p className="text-muted">or click to browse (MP4, WebM, MOV)</p>
        </div>
      )}
    </div>
  )
}
