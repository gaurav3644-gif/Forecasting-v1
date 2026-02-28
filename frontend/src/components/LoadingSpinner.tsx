import { Box, CircularProgress } from '@mui/material'

export default function LoadingSpinner() {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '60vh',
      }}
    >
      <CircularProgress sx={{ color: '#43e97b' }} />
    </Box>
  )
}
