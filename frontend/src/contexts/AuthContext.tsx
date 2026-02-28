import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import type { AuthMe } from '@/types/api'
import api from '@/api/client'

interface AuthContextValue {
  user: AuthMe | null
  loading: boolean
  refetch: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  refetch: async () => {},
})

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthMe | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchMe = async () => {
    try {
      const res = await api.get<AuthMe>('/api/auth/me')
      setUser(res.data)
    } catch {
      setUser(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchMe() }, [])

  return (
    <AuthContext.Provider value={{ user, loading, refetch: fetchMe }}>
      {children}
    </AuthContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  return useContext(AuthContext)
}
