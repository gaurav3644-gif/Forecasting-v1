import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import LoadingSpinner from '@/components/LoadingSpinner'

export default function LoginPage() {
  const { user, loading } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!loading) {
      if (user?.is_authenticated) {
        navigate('/dashboard', { replace: true })
      } else {
        // Delegate to Jinja2 signin page (handles OTP, email-only, Google OAuth)
        window.location.href = '/signin?next=%2Fdashboard'
      }
    }
  }, [user, loading, navigate])

  return <LoadingSpinner />
}
