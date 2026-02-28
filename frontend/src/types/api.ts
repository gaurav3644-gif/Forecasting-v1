export interface AuthMe {
  email: string
  is_admin: boolean
  is_authenticated: boolean
  require_approval: boolean
  auth_mode: 'otp' | 'email_only' | string
}

export interface ForecastRun {
  run_id: string
  user_email?: string
  uploaded_filename?: string | null
  grain?: string | null
  months?: number | null
  start_month?: string | null
  created_at?: string | null
  status?: string
  rows?: number | null
  skus?: number | null
}

export interface ForecastProgress {
  progress: number
  status: string
  done: boolean
  cancelled: boolean
  error?: string | null
  run_session_id?: string
}

export interface RunningForecast {
  run_session_id: string
  progress: ForecastProgress
  meta: {
    uploaded_filename?: string | null
    start_month?: string | null
    months?: number | null
    grain?: string | null
    created_at?: string | null
  }
}

export interface PendingUser {
  email: string
  created_at?: string | null
  ip?: string | null
}

export interface UserRecord {
  email: string
  status?: string
  created_at?: string | null
  approved_by?: string | null
  revoked_by?: string | null
  revoke_reason?: string | null
}

export interface DashboardData {
  runs: ForecastRun[]
  pending_users: PendingUser[]
  current_users: UserRecord[]
  revoked_users: UserRecord[]
  history_backend: string
  history_info: Record<string, unknown>
  running_forecasts: RunningForecast[]
}
