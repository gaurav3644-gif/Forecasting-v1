import { AppBar, Toolbar, Box, Button, Typography, Chip, Stack } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'

export default function TopAppBar() {
  const { user } = useAuth()
  const navigate = useNavigate()

  return (
    <AppBar position="sticky">
      <Toolbar sx={{ gap: 2, minHeight: '56px !important' }}>
        {/* Logo */}
        <Box
          component="img"
          src="/static/pitensor-logo-cropped.png"
          alt="PiTensor"
          sx={{ height: 26, cursor: 'pointer', flexShrink: 0 }}
          onClick={() => navigate('/dashboard')}
        />

        <Box sx={{ flexGrow: 1 }} />

        {user?.is_authenticated ? (
          <Stack direction="row" spacing={1} alignItems="center">
            <Button
              color="inherit"
              onClick={() => navigate('/dashboard')}
              sx={{
                color: 'rgba(255,255,255,0.82)',
                fontWeight: 700,
                fontSize: '0.85rem',
                px: 1.5,
              }}
            >
              Dashboard
            </Button>

            {user.is_admin && (
              <Chip
                label="Admin"
                size="small"
                sx={{
                  background: 'rgba(245,158,11,0.18)',
                  border: '1px solid rgba(245,158,11,0.35)',
                  color: 'rgba(245,200,60,0.95)',
                  fontWeight: 700,
                  fontSize: '0.72rem',
                }}
              />
            )}

            <Typography
              variant="caption"
              sx={{
                color: 'rgba(255,255,255,0.45)',
                display: { xs: 'none', sm: 'block' },
                maxWidth: 180,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {user.email}
            </Typography>

            <Button
              variant="outlined"
              size="small"
              onClick={() => { window.location.href = '/signout' }}
              sx={{
                color: 'rgba(255,255,255,0.75)',
                borderColor: 'rgba(255,255,255,0.28)',
                borderRadius: '999px',
                fontSize: '0.78rem',
                px: 1.5,
                '&:hover': {
                  borderColor: 'rgba(255,255,255,0.55)',
                  background: 'rgba(255,255,255,0.08)',
                },
              }}
            >
              Sign out
            </Button>
          </Stack>
        ) : (
          <Button
            variant="outlined"
            size="small"
            onClick={() => { window.location.href = '/signin' }}
            sx={{
              color: '#43e97b',
              borderColor: 'rgba(67,233,123,0.55)',
              borderRadius: '999px',
              fontWeight: 700,
              fontSize: '0.82rem',
            }}
          >
            Sign in
          </Button>
        )}
      </Toolbar>
    </AppBar>
  )
}
