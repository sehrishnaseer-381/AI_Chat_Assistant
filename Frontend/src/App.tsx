
import { useEffect, useRef, useState, type FormEvent } from 'react'
import {
  Bot,
  Eye,
  EyeOff,
  LoaderCircle,
  Mail,
  Lock,
} from 'lucide-react'
import {
  login,
  register,
  getCurrentUser,
  exchangeGoogleTicket,
} from './api'
import './App.css'
import Chat from './Chat'

type AuthMode = 'login' | 'register'

function getErrorMessage(err: unknown): string {
  if (
    typeof err === 'object' &&
    err !== null &&
    'response' in err &&
    typeof err.response === 'object' &&
    err.response !== null &&
    'data' in err.response &&
    typeof err.response.data === 'object' &&
    err.response.data !== null &&
    'detail' in err.response.data &&
    typeof err.response.data.detail === 'string'
  ) {
    return err.response.data.detail
  }

  return 'Unable to connect. Check that the backend is running.'
}

function App() {
  const [mode, setMode] = useState<AuthMode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isAuthenticated, setIsAuthenticated] = useState(
    Boolean(sessionStorage.getItem('access_token')),
  )

  const processedGoogleTicket = useRef<string | null>(null)

  // Exchange the short-lived ticket returned by the Google callback.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const ticket = params.get('google_ticket')

    if (!ticket || processedGoogleTicket.current === ticket) {
      return
    }

    processedGoogleTicket.current = ticket

    // Remove the temporary ticket from the browser URL.
    window.history.replaceState(
      {},
      document.title,
      window.location.pathname,
    )

    async function finishGoogleLogin() {
      setLoading(true)
      setError('')
      setSuccess('')

      try {
        await exchangeGoogleTicket(ticket!)

        const user = await getCurrentUser()

        sessionStorage.setItem('user_id', String(user.id))
        sessionStorage.setItem('user_email', user.email)

        setIsAuthenticated(true)
      } catch (err: unknown) {
        sessionStorage.removeItem('access_token')
        sessionStorage.removeItem('user_id')
        sessionStorage.removeItem('user_email')

        setIsAuthenticated(false)
        setError(getErrorMessage(err))
      } finally {
        setLoading(false)
      }
    }

    void finishGoogleLogin()
  }, [])

  function handleGoogleLogin() {
    setError('')
    setSuccess('')

    window.location.href =
      `${import.meta.env.VITE_API_BASE_URL}/auth/google/login`
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setSuccess('')

    const normalizedEmail = email.trim()

    if (mode === 'register' && password.length < 8) {
      setError('Password must contain at least 8 characters.')
      return
    }

    setLoading(true)

    try {
      if (mode === 'register') {
        await register(normalizedEmail, password)

        setSuccess('Account created! You can now sign in.')
        setMode('login')
        setPassword('')
        setShowPassword(false)
      } else {
        const token = await login(normalizedEmail, password)

        sessionStorage.setItem('access_token', token.access_token)

        try {
          const user = await getCurrentUser()

          sessionStorage.setItem('user_id', String(user.id))
          sessionStorage.setItem('user_email', user.email)

          setIsAuthenticated(true)
        } catch (err: unknown) {
          // Avoid leaving an invalid or incomplete local session.
          sessionStorage.removeItem('access_token')
          sessionStorage.removeItem('user_id')
          sessionStorage.removeItem('user_email')

          throw err
        }
      }
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  function switchMode() {
    setMode((currentMode) =>
      currentMode === 'login' ? 'register' : 'login',
    )
    setError('')
    setSuccess('')
    setPassword('')
    setShowPassword(false)
  }

  if (isAuthenticated) {
    return <Chat />
  }

  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="brand">
          <div className="brand-icon">
            <Bot size={27} />
          </div>
          <span>AI Assistant</span>
        </div>

        <div className="auth-heading">
          <p className="eyebrow">YOUR INTELLIGENT WORKSPACE</p>

          <h1>
            {mode === 'login'
              ? 'Welcome back'
              : 'Create your account'}
          </h1>

          <p className="subtitle">
            {mode === 'login'
              ? 'Sign in to continue your conversations.'
              : 'Create an account to get started.'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          <label htmlFor="email">Email address</label>

          <div className="input-wrap">
            <Mail size={18} aria-hidden="true" />

            <input
              id="email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              maxLength={254}
              required
              disabled={loading}
            />
          </div>

          <div className="password-label">
            <label htmlFor="password">Password</label>
          </div>

          <div className="input-wrap">
            <Lock size={18} aria-hidden="true" />

            <input
              id="password"
              type={showPassword ? 'text' : 'password'}
              placeholder={
                mode === 'register'
                  ? 'At least 8 characters'
                  : 'Your password'
              }
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete={
                mode === 'login'
                  ? 'current-password'
                  : 'new-password'
              }
              minLength={mode === 'register' ? 8 : undefined}
              maxLength={128}
              required
              disabled={loading}
            />

            <button
              type="button"
              className="visibility-button"
              onClick={() =>
                setShowPassword((visible) => !visible)
              }
              aria-label={
                showPassword ? 'Hide password' : 'Show password'
              }
              aria-pressed={showPassword}
              disabled={loading}
            >
              {showPassword ? (
                <EyeOff size={18} />
              ) : (
                <Eye size={18} />
              )}
            </button>
          </div>

          {error && (
            <p className="notice error-notice" role="alert">
              {error}
            </p>
          )}

          {success && (
            <p className="notice success-notice" role="status">
              {success}
            </p>
          )}

          <button
            type="submit"
            className="submit-button"
            disabled={loading}
          >
            {loading ? (
              <>
                <LoaderCircle
                  size={18}
                  className="spinner"
                  aria-hidden="true"
                />
                Please wait...
              </>
            ) : mode === 'login' ? (
              'Sign in'
            ) : (
              'Create account'
            )}
          </button>
        </form>

        {mode === 'login' && (
          <>
            <div className="auth-divider">
              <span>OR</span>
            </div>

            <button
              type="button"
              className="google-login-button"
              onClick={handleGoogleLogin}
              disabled={loading}
            >
              Continue with Google
            </button>
          </>
        )}

        <div className="auth-switch">
          <span>
            {mode === 'login'
              ? "Don't have an account?"
              : 'Already registered?'}
          </span>

          <button
            type="button"
            onClick={switchMode}
            disabled={loading}
          >
            {mode === 'login' ? 'Create account' : 'Sign in'}
          </button>
        </div>

        <p className="security-note">
          Your account is authenticated by your AI Assistant backend.
        </p>
      </section>

      <footer className="page-footer">
        Built for thoughtful work and better conversations.
      </footer>
    </main>
  )
}

export default App
