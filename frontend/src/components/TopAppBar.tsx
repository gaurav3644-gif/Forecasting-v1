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
                color: 'rgba(18,24,39,0.78)',
                fontWeight: 700,
                fontSize: '0.85rem',
                px: 1.5,
                '&:hover': { background: 'rgba(67,233,123,0.08)' },
              }}
            >
              Dashboard
            </Button>

            {user.is_admin && (
              <Chip
                label="Admin"
                size="small"
                sx={{
                  background: 'rgba(245,158,11,0.12)',
                  border: '1px solid rgba(245,158,11,0.30)',
                  color: 'rgba(161,99,0,0.95)',
                  fontWeight: 700,
                  fontSize: '0.72rem',
                }}
              />
            )}

            <Typography
              variant="caption"
              sx={{
                color: 'rgba(18,24,39,0.42)',
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
                color: 'rgba(18,24,39,0.65)',
                borderColor: 'rgba(18,24,39,0.20)',
                borderRadius: '999px',
                fontSize: '0.78rem',
                px: 1.5,
                '&:hover': {
                  borderColor: 'rgba(18,24,39,0.40)',
                  background: 'rgba(18,24,39,0.04)',
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
              color: '#15803d',
              borderColor: 'rgba(67,233,123,0.55)',
              borderRadius: '999px',
              fontWeight: 700,
              fontSize: '0.82rem',
              '&:hover': { background: 'rgba(67,233,123,0.08)', borderColor: '#43e97b' },
            }}
          >
            Sign in
          </Button>
        )}
      </Toolbar>
    </AppBar>
  )
}
