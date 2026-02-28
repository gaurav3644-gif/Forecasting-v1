import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Box, CssBaseline, ThemeProvider } from '@mui/material'
import { pitensorTheme } from '@/theme'
import { AuthProvider } from '@/contexts/AuthContext'
import TopAppBar from '@/components/TopAppBar'
import ProtectedRoute from '@/components/ProtectedRoute'
import LoginPage from '@/pages/LoginPage'
import DashboardPage from '@/pages/DashboardPage'

export default function App() {
  return (
    <ThemeProvider theme={pitensorTheme}>
      <CssBaseline />
      <AuthProvider>
        <BrowserRouter>
          <TopAppBar />
          <Box component="main" sx={{ minHeight: 'calc(100vh - 56px)', bgcolor: 'background.default' }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/login" element={<LoginPage />} />
              <Route
                path="/dashboard"
                element={
                  <ProtectedRoute>
                    <DashboardPage />
                  </ProtectedRoute>
                }
              />
              {/* Catch-all: future pages will be added here */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Box>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  )
}
