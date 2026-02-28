import { useState, useEffect, useRef } from 'react'
import api from '@/api/client'
import type { ForecastProgress } from '@/types/api'

interface PollOptions {
  runSessionId: string | null
  enabled: boolean
  intervalMs?: number
  onDone?: (progress: ForecastProgress) => void
}

export function useForecastPoller({
  runSessionId,
  enabled,
  intervalMs = 3000,
  onDone,
}: PollOptions) {
  const [progress, setProgress] = useState<ForecastProgress | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!enabled || !runSessionId) {
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }

    const poll = async () => {
      try {
        const res = await api.get<ForecastProgress>('/forecast_status', {
          params: { run_session_id: runSessionId },
        })
        setProgress(res.data)
        if (res.data.done || res.data.cancelled || res.data.error) {
          if (timerRef.current) clearInterval(timerRef.current)
          onDone?.(res.data)
        }
      } catch {
        // Network hiccup — keep polling
      }
    }

    poll()
    timerRef.current = setInterval(poll, intervalMs)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [runSessionId, enabled, intervalMs, onDone])

  return progress
}
