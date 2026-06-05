import { useEffect } from 'react'

export function usePolling(
  callback: () => void,
  intervalMs: number,
  enabled: boolean,
) {
  useEffect(() => {
    if (!enabled) return
    const id = window.setInterval(callback, intervalMs)
    return () => window.clearInterval(id)
  }, [callback, intervalMs, enabled])
}
