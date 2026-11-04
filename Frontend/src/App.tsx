
import { useEffect, useRef, useState, type FormEvent } from 'react'
import {
  Bot,
  Eye,
  EyeOff,
  LoaderCircle,
  Mail,
  Lock,
  MessageSquarePlus,
  PanelLeft,
  Plus,
  Search,
  Settings,
  X,
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
type AuthStep = 'email' | 'password'

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
  const [authStep, setAuthStep] = useState<AuthStep>('email')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(true)
  const [isGuestSidebarOpen, setIsGuestSidebarOpen] = useState(
    () => window.innerWidth > 640,
  )
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

  useEffect(() => {
    if (!isAuthModalOpen || loading) {
      return
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setIsAuthModalOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isAuthModalOpen, loading])

  useEffect(() => {
    function syncGuestSidebarToViewport() {
      setIsGuestSidebarOpen(window.innerWidth > 640)
    }

    window.addEventListener('resize', syncGuestSidebarToViewport)
    return () => window.removeEventListener('resize', syncGuestSidebarToViewport)
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

    if (authStep === 'email') {
      setAuthStep('password')
      return
    }

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
        setAuthStep('email')
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
    setAuthStep('email')
  }

  function openAuth(nextMode: AuthMode) {
    setMode(nextMode)
    setAuthStep('email')
    setPassword('')
    setShowPassword(false)
    setError('')
    setSuccess('')
    setIsAuthModalOpen(true)
  }

  if (isAuthenticated) {
    return <Chat />
  }

  return (
    <main
      className="guest-page"
      data-sidebar-open={isGuestSidebarOpen}
    >
      <aside
        className={`guest-sidebar ${isGuestSidebarOpen ? 'is-open' : 'is-hidden'}`}
        aria-label="Guest navigation"
      >
        <div className="guest-sidebar-header">
          <div className="guest-brand">
            <span className="guest-brand-icon"><Bot size={19} /></span>
            <span>AI Assistant</span>
          </div>
          <button
            type="button"
            className="guest-icon-button"
            aria-label="Close sidebar"
            aria-expanded={isGuestSidebarOpen}
            onClick={() => setIsGuestSidebarOpen(false)}
          >
            <PanelLeft size={19} />
          </button>
        </div>

        <nav className="guest-navigation" aria-label="Main navigation">
          <button type="button" onClick={() => openAuth('login')}>
            <MessageSquarePlus size={18} />
            <span>New chat</span>
          </button>
          <button type="button" onClick={() => openAuth('login')}>
            <Search size={18} />
            <span>Search chats</span>
          </button>
          <button type="button" onClick={() => openAuth('login')}>
            <Settings size={18} />
            <span>Settings</span>
          </button>
        </nav>

        <div className="guest-sidebar-prompt">
          <p>Get responses tailored to you</p>
          <span>Log in to save chats and continue your conversations.</span>
          <button type="button" onClick={() => openAuth('login')}>
            Log in
          </button>
        </div>
      </aside>

      <section className="guest-main">
        <header className="guest-topbar">
          <button
            type="button"
            className="guest-mobile-menu"
            onClick={() => setIsGuestSidebarOpen((open) => !open)}
            aria-label={isGuestSidebarOpen ? 'Close sidebar' : 'Open sidebar'}
            aria-expanded={isGuestSidebarOpen}
          >
            <PanelLeft size={19} />
          </button>
          <button
            type="button"
            className="guest-topbar-login"
            onClick={() => openAuth('login')}
          >
            Log in
          </button>
          <button
            type="button"
            className="guest-topbar-signup"
            onClick={() => openAuth('register')}
          >
            Sign up for free
          </button>
        </header>

        <div className="guest-content">
          <h1>What&apos;s on your mind today?</h1>
          <button
            type="button"
            className="guest-composer"
            aria-label="Message AI Assistant. Log in to start a conversation."
            onClick={() => openAuth('login')}
          >
            <Plus size={22} aria-hidden="true" />
            <span>Message AI Assistant</span>
          </button>
        </div>

        <footer className="guest-footer">
          AI Assistant can make mistakes. Check important information.
        </footer>
      </section>

      {isAuthModalOpen && (
        <div
          className="auth-overlay"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !loading) {
              setIsAuthModalOpen(false)
            }
          }}
        >
          <section
            className="auth-card login-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="auth-title"
          >
            <button
              type="button"
              className="auth-modal-close"
              onClick={() => setIsAuthModalOpen(false)}
              aria-label="Close sign-in dialog"
              disabled={loading}
            >
              <X size={20} />
            </button>

            <div className="auth-heading">
              <h1 id="auth-title">
                {authStep === 'password'
                  ? 'Enter your password'
                  : mode === 'login'
                    ? 'Log in or sign up'
                    : 'Create your account'}
              </h1>
              <p className="subtitle">
                {authStep === 'password'
                  ? `Continue with ${email.trim()}`
                  : mode === 'login'
                    ? "You'll get smarter responses and can save your conversations."
                    : 'Create an account to save your conversations and get started.'}
              </p>
            </div>

            {mode === 'login' && authStep === 'email' && (
              <>
                <button
                  type="button"
                  className="google-login-button"
                  onClick={handleGoogleLogin}
                  disabled={loading}
                >
                  <span className="google-mark" aria-hidden="true">G</span>
                  Continue with Google
                </button>

                <div className="auth-divider">
                  <span>OR</span>
                </div>
              </>
            )}

            <form onSubmit={handleSubmit} className="auth-form">
              <label htmlFor="email">Email address</label>
              <div className="input-wrap">
                <Mail size={18} aria-hidden="true" />
                <input
                  id="email"
                  type="email"
                  placeholder="Email address"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  maxLength={254}
                  required
                  disabled={loading}
                />
              </div>

              {authStep === 'password' && (
                <>
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
                          : 'Password'
                      }
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      autoComplete={
                        mode === 'login' ? 'current-password' : 'new-password'
                      }
                      minLength={mode === 'register' ? 8 : undefined}
                      maxLength={128}
                      required
                      disabled={loading}
                      autoFocus
                    />
                    <button
                      type="button"
                      className="visibility-button"
                      onClick={() => setShowPassword((visible) => !visible)}
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                      aria-pressed={showPassword}
                      disabled={loading}
                    >
                      {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                </>
              )}

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
                  authStep === 'email' ? 'Continue' : 'Log in'
                ) : (
                  authStep === 'email' ? 'Continue' : 'Create account'
                )}
              </button>
            </form>

            <div className="auth-switch">
              <span>
                {mode === 'login'
                  ? 'New to AI Assistant?'
                  : 'Already have an account?'}
              </span>
              <button
                type="button"
                onClick={switchMode}
                disabled={loading}
              >
                {mode === 'login' ? 'Sign up' : 'Log in'}
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  )
}

export default App
