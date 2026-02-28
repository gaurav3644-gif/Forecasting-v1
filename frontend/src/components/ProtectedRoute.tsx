import { useLocation } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import LoadingSpinner from './LoadingSpinner'
import type { ReactNode } from 'react'

export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) return <LoadingSpinner />

  if (!user?.is_authenticated) {
    const next = encodeURIComponent(location.pathname + location.search)
    window.location.href = `/signin?next=${next}`
    return null
  }

  return <>{children}</>
}
