
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import {
  Bot,
  Menu,
  MessageSquare,
  Plus,
  Send,
  PanelLeftClose,
  Sparkles,
  RefreshCw,
  LogOut,
  MoreHorizontal,
  Pencil,
  Check,
  X,
  Trash2,
  RotateCcw,
  Copy,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  createConversation,
  getUserConversations,
  searchConversations,
  getConversationMessages,
  renameConversation,
  deleteConversation,
  editMessage,
  regenerateAssistantMessage,
  streamChatMessage,
  retryChatMessageStream,
  type Conversation,
  type ChatMessage,
} from './api'
import './Chat.css'

export default function Chat() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [message, setMessage] = useState('')
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeConversationId, setActiveConversationId] =
    useState<number | null>(null)

  const [loadingHistory, setLoadingHistory] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [sending, setSending] = useState(false)
  const [creatingChat, setCreatingChat] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [chatError, setChatError] = useState('')

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Conversation[]>([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')

  const [openConversationMenuId, setOpenConversationMenuId] =
  useState<number | null>(null)

  const [renamingConversationId, setRenamingConversationId] =
    useState<number | null>(null)
  const [renameTitle, setRenameTitle] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isRenaming, setIsRenaming] = useState(false)

  const [failedMessage, setFailedMessage] = useState<{
  conversationId: number
  content: string
  messageId?: number
} | null>(null)
  const [editingMessageId, setEditingMessageId] = useState<number | null>(null)
  const [editedPrompt, setEditedPrompt] = useState('')
  const [editingMessageError, setEditingMessageError] = useState('')
  const [savingEditedMessage, setSavingEditedMessage] = useState(false)
  const [regeneratingMessageId, setRegeneratingMessageId] =
    useState<number | null>(null)
  const [regenerationError, setRegenerationError] = useState<{
    messageId: number
    content: string
  } | null>(null)
  const [copiedMessageId, setCopiedMessageId] = useState<number | null>(null)
  const [copyErrorMessageId, setCopyErrorMessageId] = useState<number | null>(
    null,
  )
  const [streamedResponse, setStreamedResponse] = useState('')

  const bottomRef = useRef<HTMLDivElement>(null)
  const messagesAreaRef = useRef<HTMLElement>(null)
  const shouldFollowStreamRef = useRef(true)
  const requestVersionRef = useRef(0)
  const searchRequestVersionRef = useRef(0)
  const copyFeedbackTimeoutRef = useRef<number | null>(null)
  const activeChatAbortRef = useRef<AbortController | null>(null)

  const userId = Number(sessionStorage.getItem('user_id'))
  const userEmail = sessionStorage.getItem('user_email') || 'User'

  const loadConversations = useCallback(async () => {
    if (!Number.isInteger(userId) || userId <= 0) {
      setHistoryError('Your user session is missing. Please sign in again.')
      setLoadingHistory(false)
      return
    }

    setLoadingHistory(true)
    setHistoryError('')

    try {
      const data = await getUserConversations(userId)
      setConversations(data)
    } catch {
      setHistoryError(
        'Could not load your conversations. Check the backend and retry.',
      )
    } finally {
      setLoadingHistory(false)
    }
  }, [userId])

  useEffect(() => {
    void loadConversations()
  }, [loadConversations])

  useEffect(() => {
    const messagesArea = messagesAreaRef.current
    if (!messagesArea) return

    if (sending) {
      if (!shouldFollowStreamRef.current) return

      const frame = window.requestAnimationFrame(() => {
        messagesArea.scrollTop = messagesArea.scrollHeight
      })
      return () => window.cancelAnimationFrame(frame)
    }

    bottomRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [messages, sending, streamedResponse])

  useEffect(
    () => () => {
      if (copyFeedbackTimeoutRef.current !== null) {
        window.clearTimeout(copyFeedbackTimeoutRef.current)
      }
      activeChatAbortRef.current?.abort()
    },
    [],
  )

  useEffect(() => {
    const query = searchQuery.trim()

    if (!query) return

    const requestVersion = searchRequestVersionRef.current
    let cancelled = false

    const timer = window.setTimeout(async () => {
      try {
        const results = await searchConversations(query)

        if (
          !cancelled &&
          requestVersion === searchRequestVersionRef.current
        ) {
          setSearchResults(results)
        }
      } catch (error) {
        if (
          !cancelled &&
          requestVersion === searchRequestVersionRef.current
        ) {
          setSearchError(
            'Could not search conversations. Please try again.',
          )
          setSearchResults([])
        }

        console.error('Failed to search conversations:', error)
      } finally {
        if (
          !cancelled &&
          requestVersion === searchRequestVersionRef.current
        ) {
          setSearching(false)
        }
      }
    }, 300)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [searchQuery])

  const selectConversation = async (conversationId: number) => {
    activeChatAbortRef.current?.abort()
    setStreamedResponse('')
    shouldFollowStreamRef.current = true
    const requestVersion = ++requestVersionRef.current

    setActiveConversationId(conversationId)
    setMessages([])
    setMessage('')
    setChatError('')
    setFailedMessage(null)
    setOpenConversationMenuId(null)
    setLoadingMessages(true)

    try {
      const data = await getConversationMessages(conversationId)

      // Ignore results from an older conversation request.
      if (requestVersion !== requestVersionRef.current) return

      setMessages(data)
    } catch {
      if (requestVersion !== requestVersionRef.current) return

      setChatError(
        'Could not load messages for this conversation. Please retry.',
      )
    } finally {
      if (requestVersion === requestVersionRef.current) {
        setLoadingMessages(false)
      }
    }
  }

  const handleNewChat = async () => {
    if (!Number.isInteger(userId) || userId <= 0) {
      setHistoryError('Your user session is missing. Please sign in again.')
      return
    }

    setCreatingChat(true)
    setHistoryError('')
    setChatError('')
    activeChatAbortRef.current?.abort()
    setStreamedResponse('')
    shouldFollowStreamRef.current = true

    try {
      const conversation = await createConversation(userId, 'New chat')

      setConversations((previous) => [
        conversation,
        ...previous.filter((item) => item.id !== conversation.id),
      ])

      requestVersionRef.current += 1
      setActiveConversationId(conversation.id)
      setMessages([])
      setMessage('')
      setLoadingMessages(false)
      setRenamingConversationId(null)
      setRenameTitle('')
    } catch {
      setHistoryError(
        'Could not create a new conversation. Please try again.',
      )
    } finally {
      setCreatingChat(false)
    }
  }


const handleSend = async (event?: FormEvent<HTMLFormElement>) => {
  event?.preventDefault()

  const content = message.trim()
  const conversationId = activeConversationId

  if (
    !content ||
    !conversationId ||
    sending ||
    loadingMessages ||
    regeneratingMessageId !== null
  ) return

  const requestVersion = requestVersionRef.current
  const controller = new AbortController()
  let persistedUserMessageId: number | undefined

  shouldFollowStreamRef.current = true
  setSending(true)
  setChatError('')
  setMessage('')
  setStreamedResponse('')
  setFailedMessage({ conversationId, content })
  activeChatAbortRef.current = controller

  try {
    const response = await streamChatMessage(
      conversationId,
      content,
      (userMessage) => {
        persistedUserMessageId = userMessage.id
        if (
          activeConversationId !== conversationId ||
          requestVersionRef.current !== requestVersion
        ) {
          return
        }
        setMessages((previous) => [
          ...previous.filter((item) => item.id !== userMessage.id),
          userMessage,
        ])
      },
      (chunk) => {
        if (
          activeConversationId === conversationId &&
          requestVersionRef.current === requestVersion
        ) {
          setStreamedResponse((previous) => previous + chunk)
        }
      },
      controller.signal,
    )

    // Ignore results if the conversation changed or was deleted.
    if (
      activeConversationId !== conversationId ||
      requestVersionRef.current !== requestVersion
    ) {
      return
    }

    setMessages((previous) => [
      ...previous.filter(
        (item) =>
          item.id !== response.user_message.id &&
          item.id !== response.assistant_message.id,
      ),
      response.user_message,
      response.assistant_message,
    ])
    setStreamedResponse('')
    setFailedMessage(null)

    // Refresh conversation titles and ordering.
    void loadConversations()
  } catch (error) {
    if (
      activeConversationId === conversationId &&
      requestVersionRef.current === requestVersion &&
      !controller.signal.aborted
    ) {
      setChatError('Your message could not be sent. Please try again.')
      setStreamedResponse('')
      if (persistedUserMessageId !== undefined) {
        setFailedMessage({
          conversationId,
          content,
          messageId: persistedUserMessageId,
        })
      } else {
        setMessage((current) => current || content)
      }
    }

    console.error('Failed to send chat message:', error)
  } finally {
    if (activeChatAbortRef.current === controller) {
      activeChatAbortRef.current = null
      setSending(false)
    }
  }
}

const handleRetryFailedMessage = async () => {
  const failed = failedMessage
  if (
    !failed ||
    failed.messageId === undefined ||
    failed.conversationId !== activeConversationId ||
    sending ||
    loadingMessages
  ) {
    if (failed && failed.messageId === undefined) {
      setMessage((current) => current || failed.content)
      setChatError('')
    }
    return
  }

  const requestVersion = requestVersionRef.current
  const controller = new AbortController()
  shouldFollowStreamRef.current = true
  setSending(true)
  setChatError('')
  setStreamedResponse('')
  activeChatAbortRef.current = controller

  try {
    const response = await retryChatMessageStream(
      failed.conversationId,
      failed.messageId,
      (userMessage) => {
        if (
          activeConversationId === failed.conversationId &&
          requestVersionRef.current === requestVersion
        ) {
          setMessages((previous) => [
            ...previous.filter((item) => item.id !== userMessage.id),
            userMessage,
          ])
        }
      },
      (chunk) => {
        if (
          activeConversationId === failed.conversationId &&
          requestVersionRef.current === requestVersion
        ) {
          setStreamedResponse((previous) => previous + chunk)
        }
      },
      controller.signal,
    )

    if (
      activeConversationId !== failed.conversationId ||
      requestVersionRef.current !== requestVersion
    ) {
      return
    }

    setMessages((previous) => [
      ...previous.filter(
        (item) =>
          item.id !== response.user_message.id &&
          item.id !== response.assistant_message.id,
      ),
      response.user_message,
      response.assistant_message,
    ])
    setStreamedResponse('')
    setFailedMessage(null)
  } catch (error) {
    if (
      activeConversationId === failed.conversationId &&
      requestVersionRef.current === requestVersion &&
      !controller.signal.aborted
    ) {
      setStreamedResponse('')
      setChatError('The response could not be completed. Please retry.')
    }
    console.error('Failed to retry chat response:', error)
  } finally {
    if (activeChatAbortRef.current === controller) {
      activeChatAbortRef.current = null
      setSending(false)
    }
  }
}

  const handleComposerKeyDown = (
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      void handleSend()
    }
  }

  const handleSaveEditedPrompt = async () => {
    if (
      editingMessageId === null ||
      activeConversationId === null ||
      savingEditedMessage ||
      regeneratingMessageId !== null
    ) {
      return
    }

    const conversationId = activeConversationId
    const requestVersion = requestVersionRef.current
    const nextValue = editedPrompt.trim()
    if (!nextValue) {
      setEditingMessageError('Please enter a message before saving.')
      return
    }

    setSavingEditedMessage(true)
    setEditingMessageError('')

    try {
      const response = await editMessage(
        conversationId,
        editingMessageId,
        nextValue,
      )

      if (
        activeConversationId !== conversationId ||
        requestVersionRef.current !== requestVersion
      ) {
        return
      }

      setMessages((previous) => {
        const editedMessageIndex = previous.findIndex(
          (item) => item.id === response.user_message.id,
        )

        if (editedMessageIndex === -1) return previous

        return [
          ...previous.slice(0, editedMessageIndex),
          response.user_message,
          response.assistant_message,
        ]
      })

      const updatedConversation = conversations.find(
        (item) => item.id === conversationId,
      )
      if (updatedConversation && updatedConversation.title === 'New chat') {
        setConversations((previous) =>
          previous.map((conversation) =>
            conversation.id === conversationId
              ? { ...conversation, title: response.user_message.content.trim() || conversation.title }
              : conversation,
          ),
        )
      }

      setEditingMessageId(null)
      setEditedPrompt('')
      setEditingMessageError('')
      void loadConversations()
    } catch (error) {
      console.error('Failed to edit previous prompt:', error)
      setEditingMessageError(
        'Could not update this message. Please check your input and try again.',
      )
    } finally {
      setSavingEditedMessage(false)
    }
  }

  const handleRegenerateAnswer = async (assistantMessageId: number) => {
    const conversationId = activeConversationId
    if (
      conversationId === null ||
      regeneratingMessageId !== null ||
      sending ||
      savingEditedMessage ||
      loadingMessages
    ) {
      return
    }

    const requestVersion = requestVersionRef.current
    setRegeneratingMessageId(assistantMessageId)
    setRegenerationError(null)

    try {
      const updatedMessage = await regenerateAssistantMessage(
        conversationId,
        assistantMessageId,
      )

      if (
        activeConversationId !== conversationId ||
        requestVersionRef.current !== requestVersion
      ) {
        return
      }

      setMessages((previous) =>
        previous.map((item) =>
          item.id === assistantMessageId ? updatedMessage : item,
        ),
      )
    } catch (error) {
      console.error('Failed to regenerate assistant response:', error)
      if (
        activeConversationId === conversationId &&
        requestVersionRef.current === requestVersion
      ) {
        setRegenerationError({
          messageId: assistantMessageId,
          content: 'Could not regenerate this answer. Please try again.',
        })
      }
    } finally {
      setRegeneratingMessageId(null)
    }
  }

  const handleCopyMessage = async (messageId: number, content: string) => {
    const copyWithLegacyApi = () => {
      const textArea = document.createElement('textarea')
      textArea.value = content
      textArea.setAttribute('readonly', '')
      textArea.style.position = 'fixed'
      textArea.style.opacity = '0'
      document.body.appendChild(textArea)
      textArea.select()

      try {
        if (!document.execCommand('copy')) {
          throw new Error('The browser rejected the copy command.')
        }
      } finally {
        document.body.removeChild(textArea)
      }
    }

    try {
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(content)
        } catch {
          copyWithLegacyApi()
        }
      } else {
        copyWithLegacyApi()
      }

      setCopyErrorMessageId(null)
      setCopiedMessageId(messageId)
      if (copyFeedbackTimeoutRef.current !== null) {
        window.clearTimeout(copyFeedbackTimeoutRef.current)
      }
      copyFeedbackTimeoutRef.current = window.setTimeout(() => {
        setCopiedMessageId((current) =>
          current === messageId ? null : current,
        )
        copyFeedbackTimeoutRef.current = null
      }, 1800)
    } catch (error) {
      console.error('Failed to copy assistant message:', error)
      setCopiedMessageId(null)
      setCopyErrorMessageId(messageId)
    }
  }

  const handleRenameConversation = async (conversationId: number) => {
    const title = renameTitle.trim()

    if (!title || isRenaming) return

    setIsRenaming(true)

    try {
      const updatedConversation = await renameConversation(
        conversationId,
        title,
      )

      setConversations((previous) =>
        previous.map((conversation) =>
          conversation.id === conversationId
            ? { ...conversation, title: updatedConversation.title }
            : conversation,
        ),
      )
      setSearchResults((previous) =>
        previous.map((conversation) =>
          conversation.id === conversationId
            ? { ...conversation, title: updatedConversation.title }
            : conversation,
        ),
      )

      setRenamingConversationId(null)
      setRenameTitle('')
    } catch (error) {
      console.error('Failed to rename conversation:', error)
      window.alert('Could not rename this conversation. Please try again.')
    } finally {
      setIsRenaming(false)
    }
  }


    const handleDeleteConversation = (conversationId: number) => {
      const conversation = conversations.find(
        (item) => item.id === conversationId,
      )

      if (!conversation) return

      setDeleteTarget(conversation)
    }

    const confirmDeleteConversation = async () => {
      if (!deleteTarget || isDeleting) return

      const conversationId = deleteTarget.id
      setIsDeleting(true)

      try {
        if (activeConversationId === conversationId) {
          activeChatAbortRef.current?.abort()
          setStreamedResponse('')
        }
        await deleteConversation(conversationId)

        const remainingConversations = conversations.filter(
          (item) => item.id !== conversationId,
        )

        setConversations(remainingConversations)
        setSearchResults((previous) =>
          previous.filter((item) => item.id !== conversationId),
        )

        if (renamingConversationId === conversationId) {
          setRenamingConversationId(null)
          setRenameTitle('')
        }

        if (activeConversationId === conversationId) {
          requestVersionRef.current += 1
          setMessages([])
          setMessage('')
          setChatError('')
          setLoadingMessages(false)

          if (remainingConversations.length > 0) {
            await selectConversation(remainingConversations[0].id)
          } else {
            setActiveConversationId(null)
          }
        }

        setDeleteTarget(null)
      } catch (error) {
        console.error('Failed to delete conversation:', error)
        window.alert('Could not delete this conversation. Please try again.')
      } finally {
        setIsDeleting(false)
      }
    }


  const handleLogout = () => {
    activeChatAbortRef.current?.abort()
    sessionStorage.removeItem('access_token')
    sessionStorage.removeItem('user_id')
    sessionStorage.removeItem('user_email')
    window.location.reload()
  }

  const searchActive = Boolean(searchQuery.trim())
  const visibleConversations = searchActive ? searchResults : conversations

return (
  <div className={`chat-layout ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
    <aside
      className={`chat-sidebar ${sidebarOpen ? '' : 'sidebar-hidden'}`}
    >
      <div className="sidebar-header">
        <div className="brand-mark">
          <Bot size={23} />
        </div>

        <span className="brand-name">AI Assistant</span>

        <button
          className="icon-button sidebar-toggle"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar"
          title="Close sidebar"
        >
          <PanelLeftClose size={19} />
        </button>
      </div>

      <button
        className="new-chat-button"
        onClick={() => void handleNewChat()}
        disabled={creatingChat}
      >
        <Plus size={18} />
        <span>{creatingChat ? 'Creating...' : 'New chat'}</span>
      </button>

      <div className="history-section">
        <div className="history-label">YOUR CHATS</div>

        <input
          className="conversation-search"
          type="search"
          placeholder="Search conversations..."
          aria-label="Search conversations"
          value={searchQuery}
          onChange={(event) => {
            const query = event.target.value
            searchRequestVersionRef.current += 1
            setSearchQuery(query)
            setSearchResults([])
            setSearching(Boolean(query.trim()))
            setSearchError('')
          }}
        />

        {!searchActive && loadingHistory && (
          <div className="sidebar-message">
            Loading conversations...
          </div>
        )}

        {!searchActive && historyError && (
          <div className="sidebar-error">
            <p>{historyError}</p>
            <button onClick={() => void loadConversations()}>
              Retry
            </button>
          </div>
        )}

        {!searchActive &&
          !loadingHistory &&
          !historyError &&
          conversations.length === 0 && (
            <div className="sidebar-message">
              Your conversations will appear here.
            </div>
          )}

        {searchActive && searching && (
          <div className="sidebar-message search-status" role="status">
            Searching conversations...
          </div>
        )}

        {searchActive && searchError && (
          <div className="sidebar-error search-error" role="alert">
            <p>{searchError}</p>
          </div>
        )}

        {searchActive &&
          !searching &&
          !searchError &&
          searchResults.length === 0 && (
            <div className="sidebar-message" role="status">
              No matching conversations found.
            </div>
          )}

        {visibleConversations.map((conversation) => (
          <div
            key={conversation.id}
            className={`history-item-wrapper ${
              activeConversationId === conversation.id ? 'active' : ''
            }`}
          >
            {renamingConversationId === conversation.id ? (
              <form
                className="rename-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void handleRenameConversation(conversation.id)
                }}
              >
                <input
                  autoFocus
                  value={renameTitle}
                  onChange={(event) => setRenameTitle(event.target.value)}
                  maxLength={200}
                  aria-label="Conversation title"
                  disabled={isRenaming}
                />

                <button
                  type="submit"
                  className="history-action"
                  aria-label="Save conversation name"
                  disabled={isRenaming || !renameTitle.trim()}
                >
                  <Check size={15} />
                </button>

                <button
                  type="button"
                  className="history-action"
                  aria-label="Cancel renaming"
                  onClick={() => {
                    setRenamingConversationId(null)
                    setRenameTitle('')
                  }}
                  disabled={isRenaming}
                >
                  <X size={15} />
                </button>
              </form>
            ) : (
              <>
                <button
                  type="button"
                  className={`history-item ${
                    activeConversationId === conversation.id ? 'active' : ''
                  }`}
                  onClick={() => void selectConversation(conversation.id)}
                >
                  <MessageSquare size={16} />
                  <span>{conversation.title || 'New chat'}</span>
                </button>

                <div className="conversation-actions">
                  <button
                    type="button"
                    className="history-menu-button"
                    aria-label={`Options for ${conversation.title || 'New chat'}`}
                    aria-haspopup="menu"
                    aria-expanded={openConversationMenuId === conversation.id}
                    title="Conversation options"
                    onClick={() =>
                      setOpenConversationMenuId((current) =>
                        current === conversation.id ? null : conversation.id,
                      )
                    }
                  >
                    <MoreHorizontal size={17} />
                  </button>

                  {openConversationMenuId === conversation.id && (
                    <div className="conversation-menu" role="menu">
                      <button
                        type="button"
                        className="conversation-menu-item"
                        role="menuitem"
                        onClick={() => {
                          setRenamingConversationId(conversation.id)
                          setRenameTitle(conversation.title || '')
                          setOpenConversationMenuId(null)
                        }}
                      >
                        <Pencil size={15} />
                        <span>Rename</span>
                      </button>

                      <button
                        type="button"
                        className="conversation-menu-item delete-menu-item"
                        role="menuitem"
                        onClick={() => {
                          setOpenConversationMenuId(null)
                          handleDeleteConversation(conversation.id)
                        }}
                      >
                        <Trash2 size={15} />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="user-avatar">
          {userEmail.charAt(0).toUpperCase()}
        </div>

        <div className="user-info">
          <span className="user-name">{userEmail}</span>
          <span className="user-plan">Signed in</span>
        </div>

        <button
          className="icon-button logout-button"
          onClick={handleLogout}
          title="Sign out"
          aria-label="Sign out"
        >
          <LogOut size={17} />
        </button>
      </div>
    </aside>

    <main className="chat-main">
      {!sidebarOpen && (
        <div className="topbar">
          <button
            className="icon-button"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open sidebar"
            title="Open sidebar"
          >
            <Menu size={20} />
          </button>

          <span className="topbar-title">AI Assistant</span>
        </div>
      )}

      {!activeConversationId ? (
        <div className="welcome-screen">
          <div className="welcome-icon">
            <Sparkles size={27} />
          </div>

          <h1>How can I help you today?</h1>
          <p>Ask a question, explore an idea, or work through a problem.</p>

          <button
            className="welcome-new-chat"
            onClick={() => void handleNewChat()}
            disabled={creatingChat}
          >
            <Plus size={18} />
            {creatingChat
              ? 'Creating conversation...'
              : 'Start a new conversation'}
          </button>
        </div>
      ) : (
        <>
          <header className="conversation-header">
            <div>
              <h2>
                {conversations.find(
                  (item) => item.id === activeConversationId,
                )?.title || 'Your conversation'}
              </h2>
              <span>AI Assistant</span>
            </div>

            <button
              className="icon-button"
              onClick={() => void selectConversation(activeConversationId)}
              disabled={
              loadingMessages ||
              sending ||
              savingEditedMessage ||
              regeneratingMessageId !== null
              }
              title="Reload messages"
              aria-label="Reload messages"
            >
              <RefreshCw size={17} />
            </button>
          </header>

          <section
            ref={messagesAreaRef}
            className={`messages-area ${sending ? 'streaming-active' : ''}`}
            aria-live="polite"
            onScroll={(event) => {
              const element = event.currentTarget
              const distanceFromBottom =
                element.scrollHeight - element.scrollTop - element.clientHeight
              shouldFollowStreamRef.current = distanceFromBottom < 64
            }}
          >
            {loadingMessages && (
              <div className="message-status">
                Loading messages...
              </div>
            )}

            {!loadingMessages && messages.length === 0 && !chatError && (
              <div className="empty-conversation">
                <div className="welcome-icon">
                  <Sparkles size={23} />
                </div>
                <h3>Start the conversation</h3>
                <p>Send a message below to get started.</p>
              </div>
            )}

            {messages.map((item) => (
              <article
                key={item.id}
                className={`message-row ${
                  item.role === 'user' ? 'user-row' : 'assistant-row'
                }`}
              >
                {item.role === 'assistant' && (
                  <div className="message-avatar">
                    <Bot size={17} />
                  </div>
                )}

                <div className={`message-bubble ${item.role}-bubble`}>
                  {item.role === 'user' && editingMessageId === item.id ? (
                    <div className="edit-message-box">
                      <textarea
                        value={editedPrompt}
                        onChange={(event) => setEditedPrompt(event.target.value)}
                        rows={4}
                        aria-label="Edit prompt"
                        disabled={savingEditedMessage}
                      />

                      {editingMessageError && (
                        <p className="edit-message-error">{editingMessageError}</p>
                      )}

                      <div className="edit-message-actions">
                        <button
                          type="button"
                          className="save-edit-button"
                          onClick={() => void handleSaveEditedPrompt()}
                          disabled={savingEditedMessage || !editedPrompt.trim()}
                        >
                          {savingEditedMessage ? 'Saving...' : 'Save'}
                        </button>
                        <button
                          type="button"
                          className="cancel-edit-button"
                          onClick={() => {
                            setEditingMessageId(null)
                            setEditedPrompt('')
                            setEditingMessageError('')
                          }}
                          disabled={savingEditedMessage}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="message-content">
                        {item.role === 'assistant' ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {item.content}
                          </ReactMarkdown>
                        ) : (
                          <p>{item.content}</p>
                        )}
                      </div>

                      {item.role === 'user' && (
                        <div className="message-actions">
                          <button
                            type="button"
                            className="message-action-button"
                            disabled={
                              sending ||
                              loadingMessages ||
                              savingEditedMessage ||
                              regeneratingMessageId !== null
                            }
                            onClick={() => {
                              setEditingMessageId(item.id)
                              setEditedPrompt(item.content)
                              setEditingMessageError('')
                            }}
                            aria-label="Edit message"
                          >
                            <Pencil size={14} />
                            Edit
                          </button>
                        </div>
                      )}

                      {item.role === 'assistant' && (
                        <>
                          {regenerationError?.messageId === item.id && (
                            <p className="regeneration-error" role="alert">
                              {regenerationError.content}
                            </p>
                          )}
                          {copyErrorMessageId === item.id && (
                            <p className="copy-message-error" role="alert">
                              Could not copy this answer. Try selecting and
                              copying the text.
                            </p>
                          )}
                          <div className="message-actions">
                            <button
                              type="button"
                              className="message-action-button"
                              onClick={() =>
                                void handleCopyMessage(item.id, item.content)
                              }
                              aria-label={
                                copiedMessageId === item.id
                                  ? 'Message copied'
                                  : 'Copy message'
                              }
                            >
                              {copiedMessageId === item.id ? (
                                <Check size={14} />
                              ) : (
                                <Copy size={14} />
                              )}
                              {copiedMessageId === item.id
                                ? 'Copied!'
                                : 'Copy'}
                            </button>
                            <button
                              type="button"
                              className="message-action-button"
                              onClick={() =>
                                void handleRegenerateAnswer(item.id)
                              }
                              disabled={
                                sending ||
                                loadingMessages ||
                                savingEditedMessage ||
                                regeneratingMessageId !== null
                              }
                              aria-label="Regenerate answer"
                            >
                              <RotateCcw
                                size={14}
                                className={
                                  regeneratingMessageId === item.id
                                    ? 'regenerating-icon'
                                    : undefined
                                }
                              />
                              {regeneratingMessageId === item.id
                                ? 'Regenerating...'
                                : 'Regenerate'}
                            </button>
                          </div>
                        </>
                      )}

                      <time className="message-time">
                        {new Date(item.created_at).toLocaleTimeString([], {
                          hour: 'numeric',
                          minute: '2-digit',
                        })}
                      </time>
                    </>
                  )}
                </div>
              </article>
            ))}

            {sending && (
              <div className="message-row assistant-row">
                <div className="message-avatar">
                  <Bot size={17} />
                </div>

                <div className="message-bubble assistant-bubble">
                  {streamedResponse ? (
                    <div className="message-content streaming-message">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {streamedResponse}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <div className="typing-bubble" role="status">
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </div>
                  )}
                </div>
              </div>
            )}


{chatError && (
  <div className="chat-error" role="alert">
    <p>{chatError}</p>

    {failedMessage?.conversationId === activeConversationId && (
      <button
        type="button"
        disabled={sending || loadingMessages}
        onClick={() => void handleRetryFailedMessage()}
      >
        {failedMessage.messageId !== undefined
          ? 'Retry response'
          : 'Retry message'}
      </button>
    )}

    <button
      type="button"
      disabled={loadingMessages || sending}
      onClick={() => {
        if (activeConversationId !== null) {
          void selectConversation(activeConversationId)
        }
      }}
    >
      Reload conversation
    </button>
  </div>
)}


            <div ref={bottomRef} />
          </section>

          <div className="composer-container">
            <form className="message-composer" onSubmit={handleSend}>
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Message AI Assistant..."
                rows={1}
                aria-label="Message"
              />

              <button
                type="submit"
                className="send-button"
                disabled={
                  !message.trim() ||
                  sending ||
                  loadingMessages ||
                  regeneratingMessageId !== null
                }
                aria-label="Send message"
                title="Send message"
              >
                <Send size={17} />
              </button>
            </form>

            <p className="composer-hint">
              Enter to send ┬╖ Shift + Enter for a new line
            </p>
          </div>
        </>
      )}
    </main>

    {deleteTarget && (
      <div
        className="delete-modal-backdrop"
        onClick={() => {
          if (!isDeleting) setDeleteTarget(null)
        }}
      >
        <div
          className="delete-modal"
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="delete-modal-title"
          aria-describedby="delete-modal-description"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="delete-modal-icon">
            <Trash2 size={24} />
          </div>

          <h2 id="delete-modal-title">Delete conversation?</h2>

          <p id="delete-modal-description">
            Are you sure you want to delete "
            {deleteTarget.title || 'New chat'}"?
            This action cannot be undone.
          </p>

          <div className="delete-modal-actions">
            <button
              type="button"
              className="delete-modal-cancel"
              disabled={isDeleting}
              onClick={() => setDeleteTarget(null)}
            >
              Cancel
            </button>

            <button
              type="button"
              className="delete-modal-confirm"
              disabled={isDeleting}
              onClick={() => void confirmDeleteConversation()}
            >
              {isDeleting ? 'Deleting...' : 'Delete'}
            </button>
          </div>
        </div>
      </div>
    )}
  </div>
)

}
