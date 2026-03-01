import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  IconButton,
  LinearProgress,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import RefreshIcon from '@mui/icons-material/Refresh'
import AssessmentRoundedIcon from '@mui/icons-material/AssessmentRounded'
import LightbulbRoundedIcon from '@mui/icons-material/LightbulbRounded'
import LocalShippingRoundedIcon from '@mui/icons-material/LocalShippingRounded'
import { useNavigate } from 'react-router-dom'
import api from '@/api/client'
import { useAuth } from '@/contexts/AuthContext'
import type { DashboardData, ForecastRun, PendingUser, RunningForecast, UserRecord } from '@/types/api'

// ── Running forecast card ───────────────────────────────────────────────────

function RunningForecastCard({ rf, onDone }: { rf: RunningForecast; onDone: () => void }) {
  const [pct, setPct] = useState(Math.round((rf.progress.progress ?? 0) * 100))
  const [status, setStatus] = useState(rf.progress.status ?? 'Working…')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await api.get('/forecast_status', {
          params: { run_session_id: rf.run_session_id },
        })
        const p = res.data
        setPct(Math.round((p.progress ?? 0) * 100))
        setStatus(p.status ?? '')
        if (p.done || p.cancelled || p.error) {
          if (timerRef.current) clearInterval(timerRef.current)
          onDone()
        }
      } catch { /* keep polling */ }
    }
    poll()
    timerRef.current = setInterval(poll, 2000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [rf.run_session_id, onDone])

  const handleStop = async () => {
    await api.post('/forecast_stop', null, { params: { run_session_id: rf.run_session_id } })
  }

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="flex-start" mb={1.5}>
          <Box>
            <Stack direction="row" alignItems="center" spacing={1} mb={0.3}>
              {/* Rotating gear icon */}
              <Box
                component="svg"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                sx={{
                  width: 15,
                  height: 15,
                  color: '#43e97b',
                  animation: 'spin 2s linear infinite',
                  '@keyframes spin': { from: { transform: 'rotate(0deg)' }, to: { transform: 'rotate(360deg)' } },
                }}
              >
                <path d="M12 15.5A3.5 3.5 0 0 1 8.5 12 3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5 3.5 3.5 0 0 1-3.5 3.5m7.43-2.92c.04-.32.07-.64.07-.97s-.03-.67-.07-1l2.16-1.68c.19-.15.24-.42.12-.64l-2.05-3.55c-.12-.22-.38-.3-.61-.22l-2.55 1.03c-.53-.4-1.1-.73-1.72-.98L14.5 2.42c-.04-.24-.24-.42-.5-.42h-4c-.26 0-.46.18-.49.42l-.38 2.65c-.63.25-1.2.58-1.73.98L5.4 4.65c-.23-.09-.49 0-.61.22L2.74 8.42c-.13.22-.07.49.12.64l2.16 1.68c-.04.33-.07.65-.07 1s.03.66.07.98l-2.16 1.69c-.19.15-.24.42-.12.64l2.05 3.55c.12.22.38.3.61.22l2.55-1.03c.53.4 1.1.73 1.72.98l.38 2.65c.03.24.23.42.49.42h4c.26 0 .46-.18.5-.42l.38-2.65c.62-.25 1.19-.58 1.72-.98l2.55 1.03c.23.08.49 0 .61-.22l2.05-3.55c.12-.22.07-.49-.12-.64l-2.16-1.69z" />
              </Box>
              <Typography variant="subtitle2" fontWeight={700}>Forecast running</Typography>
            </Stack>
            <Typography variant="caption" color="text.secondary">
              {rf.meta.uploaded_filename ?? 'file'} · Start: {rf.meta.start_month ?? '—'} · Months: {rf.meta.months ?? '—'}
            </Typography>
          </Box>
          <Typography variant="caption" color="text.secondary">{status}</Typography>
        </Stack>

        <LinearProgress variant="determinate" value={pct} sx={{ mb: 1 }} />

        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="caption" color="text.secondary">{pct}%</Typography>
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              variant="outlined"
              onClick={() => { window.location.href = `/forecast?run_session_id=${rf.run_session_id}` }}
            >
              View
            </Button>
            <Button size="small" variant="outlined" color="error" onClick={handleStop}>Stop</Button>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  )
}

// ── Runs table ──────────────────────────────────────────────────────────────

function RunsTable({ runs }: { runs: ForecastRun[] }) {
  if (runs.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
        No forecast runs yet. Upload a CSV above to get started.
      </Typography>
    )
  }
  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>File</TableCell>
            <TableCell>Grain</TableCell>
            <TableCell>Months</TableCell>
            <TableCell>Created</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.run_id}>
              <TableCell sx={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {run.uploaded_filename ?? '—'}
              </TableCell>
              <TableCell>{run.grain ?? '—'}</TableCell>
              <TableCell>{run.months ?? '—'}</TableCell>
              <TableCell sx={{ whiteSpace: 'nowrap' }}>
                {run.created_at ? new Date(run.created_at).toLocaleDateString() : '—'}
              </TableCell>
              <TableCell>
                <Chip label={run.status ?? 'done'} size="small" variant="outlined" />
              </TableCell>
              <TableCell>
                <Stack direction="row" spacing={0.5}>
                  <Tooltip title="Results" arrow>
                    <IconButton
                      size="small"
                      onClick={() => { window.location.href = `/results?run_session_id=${run.run_id}` }}
                      sx={{
                        color: '#43e97b',
                        background: 'rgba(67,233,123,0.10)',
                        borderRadius: '8px',
                        '&:hover': { background: 'rgba(67,233,123,0.22)', transform: 'translateY(-1px)' },
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <AssessmentRoundedIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Insights" arrow>
                    <IconButton
                      size="small"
                      onClick={() => { window.location.href = `/insights?run_session_id=${run.run_id}` }}
                      sx={{
                        color: '#f59e0b',
                        background: 'rgba(245,158,11,0.10)',
                        borderRadius: '8px',
                        '&:hover': { background: 'rgba(245,158,11,0.22)', transform: 'translateY(-1px)' },
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <LightbulbRoundedIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Supply Plan" arrow>
                    <IconButton
                      size="small"
                      onClick={() => { window.location.href = `/supply_plan?run_session_id=${run.run_id}` }}
                      sx={{
                        color: '#6366f1',
                        background: 'rgba(99,102,241,0.10)',
                        borderRadius: '8px',
                        '&:hover': { background: 'rgba(99,102,241,0.22)', transform: 'translateY(-1px)' },
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <LocalShippingRoundedIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                </Stack>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  )
}

// ── Admin sub-tables ─────────────────────────────────────────────────────────

function PendingUsersTable({ users, onAction }: { users: PendingUser[]; onAction: () => void }) {
  const approve = async (email: string) => {
    const fd = new FormData(); fd.append('email', email)
    await api.post('/admin/approve-user', fd, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })
    onAction()
  }
  if (!users.length) return <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>No pending approvals.</Typography>
  return (
    <Table size="small">
      <TableHead><TableRow><TableCell>Email</TableCell><TableCell>Requested</TableCell><TableCell>Action</TableCell></TableRow></TableHead>
      <TableBody>
        {users.map(u => (
          <TableRow key={u.email}>
            <TableCell>{u.email}</TableCell>
            <TableCell>{u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</TableCell>
            <TableCell><Button size="small" variant="contained" onClick={() => approve(u.email)}>Approve</Button></TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function CurrentUsersTable({ users, onAction }: { users: UserRecord[]; onAction: () => void }) {
  const revoke = async (email: string) => {
    const fd = new FormData(); fd.append('email', email)
    await api.post('/admin/revoke-user', fd, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })
    onAction()
  }
  if (!users.length) return <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>No users found.</Typography>
  return (
    <Table size="small">
      <TableHead><TableRow><TableCell>Email</TableCell><TableCell>Status</TableCell><TableCell>Joined</TableCell><TableCell>Action</TableCell></TableRow></TableHead>
      <TableBody>
        {users.map(u => (
          <TableRow key={u.email}>
            <TableCell>{u.email}</TableCell>
            <TableCell><Chip label={u.status ?? 'active'} size="small" variant="outlined" /></TableCell>
            <TableCell>{u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</TableCell>
            <TableCell><Button size="small" color="error" variant="outlined" onClick={() => revoke(u.email)}>Revoke</Button></TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function RevokedUsersTable({ users }: { users: UserRecord[] }) {
  if (!users.length) return <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>No revoked users.</Typography>
  return (
    <Table size="small">
      <TableHead><TableRow><TableCell>Email</TableCell><TableCell>Revoked At</TableCell><TableCell>Reason</TableCell></TableRow></TableHead>
      <TableBody>
        {users.map(u => (
          <TableRow key={u.email}>
            <TableCell>{u.email}</TableCell>
            <TableCell>{u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</TableCell>
            <TableCell color="text.secondary">{u.revoke_reason ?? '—'}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

// ── Main Dashboard page ──────────────────────────────────────────────────────

export default function DashboardPage() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState(0)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const fetchDashboard = useCallback(async () => {
    try {
      const res = await api.get<DashboardData>('/api/dashboard')
      setData(res.data)
    } catch (e) {
      console.error('Dashboard fetch error', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchDashboard() }, [fetchDashboard])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadError(null)
    setUploading(true)
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await api.post<{ run_session_id: string }>('/upload', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      // Phase 2 will add a React /forecast route. For now use Jinja2 page.
      window.location.href = `/forecast?run_session_id=${res.data.run_session_id ?? ''}`
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setUploadError(detail ?? 'Upload failed — check your CSV has: date, item, store, sales.')
    } finally {
      setUploading(false)
      // Reset file input
      e.target.value = ''
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <CircularProgress sx={{ color: '#43e97b' }} />
      </Box>
    )
  }

  const runs = data?.runs ?? []
  const pending = data?.pending_users ?? []
  const current = data?.current_users ?? []
  const revoked = data?.revoked_users ?? []
  const running = data?.running_forecasts ?? []
  const isAdmin = user?.is_admin ?? false

  return (
    <Box sx={{ maxWidth: 1480, mx: 'auto', px: { xs: 2, md: 4 }, py: 3 }}>

      {/* Hero */}
      <Card
        sx={{
          mb: 3,
          background: `
            radial-gradient(1200px 420px at 8% -10%, rgba(67,233,123,0.20), transparent 60%),
            radial-gradient(800px 300px at 92% -10%, rgba(56,249,215,0.16), transparent 55%),
            #fff
          `,
        }}
      >
        <CardContent sx={{ p: { xs: 2.5, md: 3 } }}>
          <Stack direction="row" justifyContent="space-between" alignItems="flex-start" flexWrap="wrap" gap={2}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>PiTensor</Typography>
              <Typography variant="h4" sx={{ mb: 0.5 }}>Dashboard</Typography>
              <Typography variant="body2" color="text.secondary">
                Run new forecasts and review your saved history.
              </Typography>
            </Box>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
              {isAdmin && (
                <Chip label="Admin" size="small"
                  sx={{ background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.35)', color: '#92400e', fontWeight: 700 }} />
              )}
              <Chip label={`DB: ${data?.history_backend ?? '—'}`} size="small" variant="outlined"
                sx={{ color: 'text.secondary', borderColor: 'rgba(18,24,39,0.15)' }} />
              <Button size="small" variant="outlined" startIcon={<RefreshIcon />} onClick={fetchDashboard}>
                Refresh
              </Button>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {/* Running forecasts */}
      {running.map(rf => (
        <RunningForecastCard key={rf.run_session_id} rf={rf} onDone={fetchDashboard} />
      ))}

      {/* Upload new forecast */}
      <Card sx={{ mb: 3 }}>
        <CardContent sx={{ p: { xs: 2, md: 2.5 } }}>
          <Typography variant="h6" sx={{ mb: 0.5 }}>New Forecast</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Upload a CSV with columns: <code>date</code>, <code>item</code>, <code>store</code>, <code>sales</code>
          </Typography>
          <Button
            variant="contained"
            component="label"
            startIcon={uploading ? <CircularProgress size={16} sx={{ color: '#0b1410' }} /> : <UploadFileIcon />}
            disabled={uploading}
          >
            {uploading ? 'Uploading…' : 'Upload CSV & Configure Forecast'}
            <input type="file" accept=".csv,.xlsx,.xls" hidden onChange={handleUpload} />
          </Button>
          {uploadError && (
            <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }} onClose={() => setUploadError(null)}>
              {uploadError}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Runs history / Admin tabs */}
      <Card>
        <CardContent sx={{ p: { xs: 2, md: 2.5 } }}>
          {isAdmin ? (
            <>
              <Tabs
                value={tab}
                onChange={(_, v: number) => setTab(v)}
                sx={{ mb: 2, borderBottom: '1px solid rgba(18,24,39,0.08)' }}
                TabIndicatorProps={{ style: { background: '#43e97b' } }}
              >
                <Tab label={`Forecast Runs (${runs.length})`} />
                <Tab label={`Pending Users (${pending.length})`} />
                <Tab label={`Current Users (${current.length})`} />
                <Tab label={`Revoked (${revoked.length})`} />
              </Tabs>
              {tab === 0 && <RunsTable runs={runs} />}
              {tab === 1 && <PendingUsersTable users={pending} onAction={fetchDashboard} />}
              {tab === 2 && <CurrentUsersTable users={current} onAction={fetchDashboard} />}
              {tab === 3 && <RevokedUsersTable users={revoked} />}
            </>
          ) : (
            <>
              <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">Forecast History ({runs.length})</Typography>
              </Stack>
              <RunsTable runs={runs} />
            </>
          )}
        </CardContent>
      </Card>

      {/* Navigate to Data page */}
      <Box sx={{ mt: 2, textAlign: 'right' }}>
        <Button variant="text" size="small" color="inherit"
          sx={{ color: 'text.secondary', fontSize: '0.78rem' }}
          onClick={() => { window.location.href = '/data' }}>
          Manage saved datasets →
        </Button>
      </Box>
    </Box>
  )
}
