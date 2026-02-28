import { createTheme } from '@mui/material/styles'

const BRAND_GREEN = '#43e97b'
const BRAND_CYAN = '#38f9d7'
const NAVBAR_BG = '#0a0f0d'

export const pitensorTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: BRAND_GREEN, contrastText: '#0b1410' },
    secondary: { main: BRAND_CYAN, contrastText: '#0b1410' },
    background: { default: '#f7f8fc', paper: '#ffffff' },
    text: {
      primary: 'rgba(18,24,39,0.90)',
      secondary: 'rgba(18,24,39,0.62)',
    },
  },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif',
    h4: { fontWeight: 800, letterSpacing: '-0.03em' },
    h5: { fontWeight: 700, letterSpacing: '-0.02em' },
    h6: { fontWeight: 700, letterSpacing: '-0.01em' },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: {
      styleOverrides: {
        containedPrimary: {
          background: `linear-gradient(90deg, ${BRAND_GREEN} 0%, ${BRAND_CYAN} 100%)`,
          borderRadius: '999px',
          color: '#0b1410',
          fontWeight: 700,
          boxShadow: '0 8px 22px rgba(67,233,123,0.22)',
          textTransform: 'none',
          '&:hover': {
            background: `linear-gradient(90deg, ${BRAND_GREEN} 0%, ${BRAND_CYAN} 100%)`,
            boxShadow: '0 12px 30px rgba(67,233,123,0.30)',
            transform: 'translateY(-1px)',
          },
        },
        outlinedPrimary: {
          borderRadius: '999px',
          borderColor: 'rgba(67,233,123,0.55)',
          color: '#15803d',
          fontWeight: 600,
          textTransform: 'none',
          '&:hover': {
            background: `linear-gradient(90deg, ${BRAND_GREEN} 0%, ${BRAND_CYAN} 100%)`,
            borderColor: 'transparent',
            color: '#0b1410',
          },
        },
        outlinedSecondary: {
          borderRadius: '999px',
          textTransform: 'none',
        },
        text: { textTransform: 'none' },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: NAVBAR_BG,
          borderBottom: '1px solid rgba(67,233,123,0.13)',
          boxShadow: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '18px',
          border: '1px solid rgba(18,24,39,0.08)',
          boxShadow: '0 10px 32px rgba(18,24,39,0.06)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: '999px', fontWeight: 600 },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        bar: {
          background: `linear-gradient(90deg, ${BRAND_GREEN} 0%, ${BRAND_CYAN} 100%)`,
        },
        root: {
          borderRadius: '999px',
          backgroundColor: 'rgba(18,24,39,0.10)',
          height: 8,
        },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: { '& .MuiTableCell-head': { fontWeight: 700, color: 'rgba(18,24,39,0.65)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.04em' } },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: { '&:hover': { backgroundColor: 'rgba(67,233,123,0.04)' } },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: { textTransform: 'none', fontWeight: 600 },
      },
    },
  },
})
