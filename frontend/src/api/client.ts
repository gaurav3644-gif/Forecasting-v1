import axios from 'axios'

// In development: Vite proxy forwards all requests to FastAPI at :8000.
// In production:  React is served by FastAPI at the same origin.
// The fa_user HMAC cookie is sent automatically on every request.
const api = axios.create({
  baseURL: '/',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const status = err.response?.status
    if (status === 401 || status === 403) {
      const next = encodeURIComponent(window.location.pathname + window.location.search)
      window.location.href = `/signin?next=${next}`
    }
    return Promise.reject(err)
  },
)

export default api
