
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import {
  Bot,
  ArrowDown,
  Camera,
  Image as ImageIcon,
  Moon,
  MessageSquare,
  Search,
  Plus,
  Send,
  PanelLeftClose,
  PanelLeftOpen,
  SquarePen,
  Sparkles,
  LogOut,
  MoreHorizontal,
  Pencil,
  Check,
  X,
  Trash2,
  RotateCcw,
  Copy,
  Sun,
  Share2,
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
  type ModelPreference,
} from './api'
import './Chat.css'

type PendingAttachment = {
  id: string
  file: File
  previewUrl: string
}

const MAX_IMAGE_COUNT = 3
const MAX_IMAGE_SIZE_BYTES = 4 * 1024 * 1024
const MAX_TOTAL_IMAGE_SIZE_BYTES = MAX_IMAGE_COUNT * MAX_IMAGE_SIZE_BYTES
const SUPPORTED_IMAGE_TYPES = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
])

function isModelPreference(value: string): value is ModelPreference {
  return value === 'auto' ||
    value === 'openai/gpt-oss-20b' ||
    value === 'openai/gpt-oss-120b'
}

function getApiErrorMessage(error: unknown, fallback: string): string {
  if (
    typeof error === 'object' &&
    error !== null &&
    'response' in error &&
    typeof error.response === 'object' &&
    error.response !== null &&
    'data' in error.response &&
    typeof error.response.data === 'object' &&
    error.response.data !== null &&
    'detail' in error.response.data &&
    typeof error.response.data.detail === 'string'
  ) {
    return error.response.data.detail
  }

  return fallback
}

async function copyTextToClipboard(text: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return
    } catch {
      // Fall through to the legacy clipboard API for restricted browser contexts.
    }
  }

  const textArea = document.createElement('textarea')
  textArea.value = text
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

export default function Chat() {
  const [sidebarOpen, setSidebarOpen] = useState(
    () => window.innerWidth > 800,
  )
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const savedTheme = localStorage.getItem('chat-theme')
    return savedTheme === 'light' ? 'light' : 'dark'
  })
  const [message, setMessage] = useState('')
  const [pendingAttachments, setPendingAttachments] = useState<
    PendingAttachment[]
  >([])
  const [attachmentMenuOpen, setAttachmentMenuOpen] = useState(false)
  const [attachmentError, setAttachmentError] = useState('')
  const [modelPreference, setModelPreference] = useState<ModelPreference>(() => {
    const savedPreference = localStorage.getItem('model-preference')
    return savedPreference && isModelPreference(savedPreference)
      ? savedPreference
      : 'auto'
  })
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeConversationId, setActiveConversationId] =
    useState<number | null>(null)

  const [loadingHistory, setLoadingHistory] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [showScrollToBottom, setShowScrollToBottom] = useState(false)
  const [sending, setSending] = useState(false)
  const [creatingChat, setCreatingChat] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [chatError, setChatError] = useState('')

  const [searchQuery, setSearchQuery] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchResults, setSearchResults] = useState<Conversation[]>([])
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')

  const [openConversationMenuId, setOpenConversationMenuId] =
  useState<number | null>(null)
  const [chatActionsOpen, setChatActionsOpen] = useState(false)
  const [headerFeedback, setHeaderFeedback] = useState('')

  const [renamingConversationId, setRenamingConversationId] =
    useState<number | null>(null)
  const [renameTitle, setRenameTitle] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isRenaming, setIsRenaming] = useState(false)

  const [failedMessage, setFailedMessage] = useState<{
  conversationId: number
  content: string
  modelPreference: ModelPreference
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
  const composerInputRef = useRef<HTMLTextAreaElement>(null)
  const shouldFollowStreamRef = useRef(true)
  const requestVersionRef = useRef(0)
  const searchRequestVersionRef = useRef(0)
  const copyFeedbackTimeoutRef = useRef<number | null>(null)
  const activeChatAbortRef = useRef<AbortController | null>(null)
  const chatActionsRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const headerFeedbackTimeoutRef = useRef<number | null>(null)
  const initialConversationHandledRef = useRef(false)
  const pendingAttachmentsRef = useRef<PendingAttachment[]>([])
  const imageInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const attachmentMenuRef = useRef<HTMLDivElement>(null)

  const userId = Number(sessionStorage.getItem('user_id'))
  const userEmail = sessionStorage.getItem('user_email') || 'User'

  useLayoutEffect(() => {
    const textarea = composerInputRef.current
    if (!textarea) return

    textarea.style.height = '0px'
    const maxHeight = 240
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`
    textarea.style.overflowY =
      textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [message])

  const clearPendingAttachments = useCallback(() => {
    pendingAttachmentsRef.current.forEach((attachment) =>
      URL.revokeObjectURL(attachment.previewUrl),
    )
    pendingAttachmentsRef.current = []
    setPendingAttachments([])
  }, [])

  const handleAttachmentSelection = (files: FileList | null) => {
    if (!files?.length) return

    const selected = Array.from(files)
    const current = pendingAttachmentsRef.current
    let totalSize = current.reduce(
      (total, attachment) => total + attachment.file.size,
      0,
    )
    const next = [...current]
    let error = ''

    for (const file of selected) {
      if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
        error = 'Choose a PNG, JPG, JPEG, or WEBP image.'
        continue
      }
      if (file.size > MAX_IMAGE_SIZE_BYTES) {
        error = 'Each image must be 4 MB or smaller.'
        continue
      }
      if (next.length >= MAX_IMAGE_COUNT) {
        error = 'Attach no more than 3 images.'
        break
      }
      if (totalSize + file.size > MAX_TOTAL_IMAGE_SIZE_BYTES) {
        error = 'The combined image size cannot exceed 12 MB.'
        continue
      }

      totalSize += file.size
      next.push({
        id: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
      })
    }

    pendingAttachmentsRef.current = next
    setPendingAttachments(next)
    setAttachmentError(error)
  }

  const removePendingAttachment = (id: string) => {
    const removed = pendingAttachmentsRef.current.find(
      (attachment) => attachment.id === id,
    )
    if (removed) URL.revokeObjectURL(removed.previewUrl)
    const next = pendingAttachmentsRef.current.filter(
      (attachment) => attachment.id !== id,
    )
    pendingAttachmentsRef.current = next
    setPendingAttachments(next)
    setAttachmentError('')
  }

  useEffect(() => {
    localStorage.setItem('chat-theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('model-preference', modelPreference)
  }, [modelPreference])

  useEffect(
    () => () => {
      pendingAttachmentsRef.current.forEach((attachment) =>
        URL.revokeObjectURL(attachment.previewUrl),
      )
    },
    [],
  )

  useEffect(() => {
    if (!attachmentMenuOpen) return

    const closeOnOutsideClick = (event: PointerEvent) => {
      if (
        event.target instanceof Node &&
        !attachmentMenuRef.current?.contains(event.target)
      ) {
        setAttachmentMenuOpen(false)
      }
    }

    document.addEventListener('pointerdown', closeOnOutsideClick)
    return () =>
      document.removeEventListener('pointerdown', closeOnOutsideClick)
  }, [attachmentMenuOpen])

  useEffect(() => {
    const mobileLayout = window.matchMedia('(max-width: 800px)')
    const syncSidebarToViewport = (event: MediaQueryListEvent) => {
      setSidebarOpen(!event.matches)
    }

    mobileLayout.addEventListener('change', syncSidebarToViewport)
    return () =>
      mobileLayout.removeEventListener('change', syncSidebarToViewport)
  }, [])

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
      if (headerFeedbackTimeoutRef.current !== null) {
        window.clearTimeout(headerFeedbackTimeoutRef.current)
      }
      activeChatAbortRef.current?.abort()
    },
    [],
  )

  useEffect(() => {
    if (!chatActionsOpen) return

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (
        event.target instanceof Node &&
        !chatActionsRef.current?.contains(event.target)
      ) {
        setChatActionsOpen(false)
      }
    }
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setChatActionsOpen(false)
    }

    document.addEventListener('pointerdown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsideClick)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [chatActionsOpen])

  const closeConversationSearch = useCallback(() => {
    searchRequestVersionRef.current += 1
    setSearchOpen(false)
    setSearchQuery('')
    setSearchResults([])
    setSearching(false)
    setSearchError('')
  }, [])

  useEffect(() => {
    if (!searchOpen) return

    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') closeConversationSearch()
    }

    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [closeConversationSearch, searchOpen])

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

  const selectConversation = useCallback(async (conversationId: number) => {
    activeChatAbortRef.current?.abort()
    clearPendingAttachments()
    setAttachmentError('')
    setStreamedResponse('')
    setShowScrollToBottom(false)
    shouldFollowStreamRef.current = true
    const requestVersion = ++requestVersionRef.current

    setActiveConversationId(conversationId)
    setMessages([])
    setMessage('')
    setAttachmentError('')
    setChatError('')
    setFailedMessage(null)
    setOpenConversationMenuId(null)
    setSearchOpen(false)
    setSearchQuery('')
    setSearchResults([])
    setLoadingMessages(true)
    const url = new URL(window.location.href)
    url.searchParams.set('conversation', String(conversationId))
    window.history.replaceState({}, '', url)

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
  }, [clearPendingAttachments])

  const toggleConversationSearch = () => {
    if (searchOpen) {
      closeConversationSearch()
      return
    }

    setSearchOpen(true)
    window.requestAnimationFrame(() => searchInputRef.current?.focus())
  }

  useEffect(() => {
    if (loadingHistory || initialConversationHandledRef.current) return

    initialConversationHandledRef.current = true
    const requestedId = Number(
      new URLSearchParams(window.location.search).get('conversation'),
    )
    if (
      Number.isInteger(requestedId) &&
      conversations.some((conversation) => conversation.id === requestedId)
    ) {
      void selectConversation(requestedId)
    }
  }, [conversations, loadingHistory, selectConversation])

  const handleNewChat = async () => {
    clearPendingAttachments()
    setAttachmentError('')
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
  let conversationId = activeConversationId

  if (
    (!content && pendingAttachmentsRef.current.length === 0) ||
    creatingChat ||
    sending ||
    loadingMessages ||
    regeneratingMessageId !== null
  ) return

  if (!conversationId) {
    if (!Number.isInteger(userId) || userId <= 0) {
      setHistoryError('Your user session is missing. Please sign in again.')
      return
    }

    setCreatingChat(true)
    setHistoryError('')
    try {
      const conversation = await createConversation(userId, 'New chat')
      conversationId = conversation.id
      setConversations((previous) => [
        conversation,
        ...previous.filter((item) => item.id !== conversation.id),
      ])
      requestVersionRef.current += 1
      setActiveConversationId(conversation.id)
      setMessages([])
      setLoadingMessages(false)
      setRenamingConversationId(null)
      setRenameTitle('')
    } catch {
      setHistoryError(
        'Could not create a new conversation. Please try again.',
      )
      return
    } finally {
      setCreatingChat(false)
    }
  }

  const requestVersion = requestVersionRef.current
  const controller = new AbortController()
  const attachmentsToSend = [...pendingAttachmentsRef.current]
  let persistedUserMessageId: number | undefined

  shouldFollowStreamRef.current = true
  setSending(true)
  setChatError('')
  setMessage('')
  setStreamedResponse('')
  setFailedMessage({ conversationId, content, modelPreference })
  activeChatAbortRef.current = controller

  try {
    const response = await streamChatMessage(
      conversationId,
      content,
      (userMessage) => {
        persistedUserMessageId = userMessage.id
        clearPendingAttachments()
        setAttachmentError('')
        if (requestVersionRef.current !== requestVersion) {
          return
        }
        setMessages((previous) => [
          ...previous.filter((item) => item.id !== userMessage.id),
          userMessage,
        ])
      },
      (chunk) => {
        if (requestVersionRef.current === requestVersion) {
          setStreamedResponse((previous) => previous + chunk)
        }
      },
      controller.signal,
      modelPreference,
      () => {
        if (requestVersionRef.current === requestVersion) {
          setStreamedResponse('')
        }
      },
      attachmentsToSend.map((attachment) => attachment.file),
    )

    // Ignore results if the conversation changed or was deleted.
    if (requestVersionRef.current !== requestVersion) {
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
    clearPendingAttachments()

    // Refresh conversation titles and ordering.
    void loadConversations()
  } catch (error) {
    if (
      requestVersionRef.current === requestVersion &&
      !controller.signal.aborted
    ) {
      setChatError('Your message could not be sent. Please try again.')
      setStreamedResponse('')
      if (persistedUserMessageId !== undefined) {
        setFailedMessage({
          conversationId,
          content,
          modelPreference,
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
      failed.modelPreference,
      () => {
        if (
          activeConversationId === failed.conversationId &&
          requestVersionRef.current === requestVersion
        ) {
          setStreamedResponse('')
        }
      },
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

    const originalMessage = messages.find(
      (item) => item.id === editingMessageId,
    )
    if (!originalMessage) {
      setEditingMessageError('This message is no longer available.')
      return
    }
    if (nextValue === originalMessage.content.trim()) {
      setEditingMessageId(null)
      setEditedPrompt('')
      setEditingMessageError('')
      return
    }

    setSavingEditedMessage(true)
    setEditingMessageError('')

    try {
      const response = await editMessage(
        conversationId,
        editingMessageId,
        nextValue,
        modelPreference,
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
        getApiErrorMessage(
          error,
          'Could not update this message. Please try again.',
        ),
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
        modelPreference,
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
    try {
      await copyTextToClipboard(content)

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
      console.error('Failed to copy message:', error)
      setCopiedMessageId(null)
      setCopyErrorMessageId(messageId)
    }
  }

  const handleSharePrompt = async (content: string) => {
    try {
      if (navigator.share) {
        await navigator.share({ text: content })
        return
      }

      await copyTextToClipboard(content)
      showHeaderFeedback('Prompt copied to clipboard')
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      console.error('Failed to share prompt:', error)
      showHeaderFeedback('Could not share prompt')
    }
  }

  const showHeaderFeedback = (message: string) => {
    setHeaderFeedback(message)
    if (headerFeedbackTimeoutRef.current !== null) {
      window.clearTimeout(headerFeedbackTimeoutRef.current)
    }
    headerFeedbackTimeoutRef.current = window.setTimeout(() => {
      setHeaderFeedback('')
      headerFeedbackTimeoutRef.current = null
    }, 2200)
  }

  const handleShareConversation = async () => {
    if (activeConversationId === null) return

    const url = new URL(window.location.href)
    url.searchParams.set('conversation', String(activeConversationId))
    const title =
      conversations.find((item) => item.id === activeConversationId)?.title ||
      'AI Assistant conversation'

    try {
      if (navigator.share) {
        try {
          await navigator.share({
            title,
            text: 'Open this conversation in AI Assistant.',
            url: url.toString(),
          })
          showHeaderFeedback('Shared successfully')
          return
        } catch (error) {
          if (error instanceof DOMException && error.name === 'AbortError') {
            return
          }
        }
      }

      await copyTextToClipboard(url.toString())
      showHeaderFeedback('Conversation link copied')
    } catch (error) {
      console.error('Failed to share conversation:', error)
      showHeaderFeedback('Could not share conversation')
    }
  }

  const handleCopyConversation = async () => {
    const transcript = messages
      .map((item) => `${item.role === 'user' ? 'You' : 'AI Assistant'}:\n${item.content}`)
      .join('\n\n')

    if (!transcript) {
      showHeaderFeedback('There are no messages to copy')
      setChatActionsOpen(false)
      return
    }

    try {
      await copyTextToClipboard(transcript)
      showHeaderFeedback('Conversation copied')
    } catch (error) {
      console.error('Failed to copy conversation:', error)
      showHeaderFeedback('Could not copy conversation')
    } finally {
      setChatActionsOpen(false)
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
        const conversation =
          conversations.find((item) => item.id === conversationId) ??
          searchResults.find((item) => item.id === conversationId)

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
          const url = new URL(window.location.href)
          url.searchParams.delete('conversation')
          window.history.replaceState({}, '', url)
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
  const searchItems = searchActive ? searchResults : conversations.slice(0, 6)

return (
  <div
    className={`chat-layout ${sidebarOpen ? '' : 'sidebar-collapsed'}`}
    data-theme={theme}
  >
    {sidebarOpen && (
      <button
        type="button"
        className="sidebar-backdrop"
        aria-label="Close navigation"
        onClick={() => setSidebarOpen(false)}
      />
    )}

    <aside
      className={`chat-sidebar ${sidebarOpen ? '' : 'sidebar-hidden'}`}
    >
      <div className="sidebar-header">
        <div className="brand-mark">
          <Bot size={23} />
        </div>

        <span className="brand-name">AI Assistant</span>

        <button
          type="button"
          className={`icon-button sidebar-search-toggle ${searchOpen ? 'active' : ''}`}
          onClick={toggleConversationSearch}
          aria-label={searchOpen ? 'Close conversation search' : 'Search conversations'}
          aria-expanded={searchOpen}
          aria-controls="conversation-search-dialog"
          title={searchOpen ? 'Close search' : 'Search conversations'}
        >
          {searchOpen ? <X size={18} /> : <Search size={18} />}
        </button>

        <button
          type="button"
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
        {loadingHistory && (
          <div className="sidebar-message">
            Loading conversations...
          </div>
        )}

        {historyError && (
          <div className="sidebar-error">
            <p>{historyError}</p>
            <button onClick={() => void loadConversations()}>
              Retry
            </button>
          </div>
        )}

        {!loadingHistory &&
          !historyError &&
          conversations.length === 0 && (
            <div className="sidebar-message">
              Your conversations will appear here.
            </div>
          )}

        {conversations.map((conversation) => (
          <div
            key={conversation.id}
            className={`history-item-wrapper ${
              activeConversationId === conversation.id ? 'active' : ''
            }`}
          >
            {renamingConversationId === conversation.id &&
            activeConversationId !== conversation.id ? (
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
        <button
          type="button"
          className="icon-button theme-toggle"
          onClick={() =>
            setTheme((current) => current === 'dark' ? 'light' : 'dark')
          }
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </button>

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

    {searchOpen && (
      <div
        className="conversation-search-backdrop"
        onClick={toggleConversationSearch}
      >
        <section
          id="conversation-search-dialog"
          className="conversation-search-dialog"
          role="dialog"
          aria-modal="true"
          aria-label="Search chats"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="conversation-search-header">
            <input
              ref={searchInputRef}
              className="conversation-search-input"
              type="search"
              placeholder="Search..."
              aria-label="Search chats"
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
            <button
              type="button"
              className="conversation-search-close"
              aria-label="Close search"
              onClick={toggleConversationSearch}
            >
              <X size={20} />
            </button>
          </div>

          <div className="conversation-search-results">
            <div className="conversation-search-label">
              {searchActive ? 'Search results' : 'Recent chats'}
            </div>

            {searchActive && searching && (
              <p className="conversation-search-message" role="status">
                Searching...
              </p>
            )}
            {searchActive && searchError && (
              <p className="conversation-search-message is-error" role="alert">
                {searchError}
              </p>
            )}
            {searchActive &&
              !searching &&
              !searchError &&
              searchItems.length === 0 && (
                <p className="conversation-search-message" role="status">
                  No matching conversations found.
                </p>
              )}
            {!searchActive && loadingHistory && (
              <p className="conversation-search-message" role="status">
                Loading conversations...
              </p>
            )}
            {!searchActive && historyError && (
              <p className="conversation-search-message is-error" role="alert">
                {historyError}{' '}
                <button
                  type="button"
                  className="conversation-search-retry"
                  onClick={() => void loadConversations()}
                >
                  Retry
                </button>
              </p>
            )}
            {!searchActive &&
              !historyError &&
              !loadingHistory &&
              conversations.length === 0 && (
                <p className="conversation-search-message">
                  Your conversations will appear here.
                </p>
              )}

            {(searchActive
              ? !searching && !searchError
              : !historyError) &&
              searchItems.map((conversation) => (
                <button
                  type="button"
                  key={conversation.id}
                  className="conversation-search-result"
                  onClick={() => void selectConversation(conversation.id)}
                >
                  <MessageSquare size={20} aria-hidden="true" />
                  <span>{conversation.title || 'New chat'}</span>
                </button>
              ))}
          </div>
        </section>
      </div>
    )}

    <main className="chat-main">
      <header className="topbar chat-topbar">
        <div className="topbar-leading">
          {!sidebarOpen && (
            <>
              <button
                type="button"
                className="icon-button"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
                title="Open sidebar"
              >
                <PanelLeftOpen size={20} />
              </button>
              <button
                type="button"
                className="icon-button"
                onClick={() => void handleNewChat()}
                aria-label="New chat"
                title="New chat"
                disabled={creatingChat}
              >
                <SquarePen size={19} />
              </button>
            </>
          )}
        </div>

        {activeConversationId !== null && (
          <div className="chat-header-actions">
            {renamingConversationId === activeConversationId ? (
              <form
                className="header-rename-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void handleRenameConversation(activeConversationId)
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
                  className="icon-button"
                  aria-label="Save conversation name"
                  disabled={isRenaming || !renameTitle.trim()}
                >
                  <Check size={17} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Cancel renaming"
                  onClick={() => {
                    setRenamingConversationId(null)
                    setRenameTitle('')
                  }}
                  disabled={isRenaming}
                >
                  <X size={17} />
                </button>
              </form>
            ) : (
              <>
                <div className="chat-actions-wrap" ref={chatActionsRef}>
                  <button
                    type="button"
                    className="icon-button chat-actions-button"
                    onClick={() => setChatActionsOpen((open) => !open)}
                    aria-label="Conversation options"
                    aria-haspopup="menu"
                    aria-expanded={chatActionsOpen}
                    title="Conversation options"
                  >
                    <MoreHorizontal size={21} />
                  </button>

                  {chatActionsOpen && (
                    <div className="conversation-menu chat-actions-menu" role="menu">
                      <button
                        type="button"
                        className="conversation-menu-item"
                        role="menuitem"
                        onClick={() => {
                          const conversation = conversations.find(
                            (item) => item.id === activeConversationId,
                          )
                          setRenameTitle(conversation?.title || '')
                          setRenamingConversationId(activeConversationId)
                          setChatActionsOpen(false)
                        }}
                      >
                        <Pencil size={15} />
                        <span>Rename</span>
                      </button>
                      <button
                        type="button"
                        className="conversation-menu-item"
                        role="menuitem"
                        onClick={() => void handleCopyConversation()}
                      >
                        <Copy size={15} />
                        <span>Copy conversation</span>
                      </button>
                      <button
                        type="button"
                        className="conversation-menu-item delete-menu-item"
                        role="menuitem"
                        onClick={() => {
                          setChatActionsOpen(false)
                          handleDeleteConversation(activeConversationId)
                        }}
                      >
                        <Trash2 size={15} />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>

                <button
                  type="button"
                  className="icon-button header-share-button"
                  onClick={() => void handleShareConversation()}
                  aria-label="Share conversation"
                  title="Share conversation"
                >
                  <Share2 size={17} />
                  <span>Share</span>
                </button>
              </>
            )}
          </div>
        )}

        {headerFeedback && (
          <span className="header-feedback" role="status">
            {headerFeedback}
          </span>
        )}
      </header>

      <div
        className={`chat-content ${activeConversationId === null ? 'new-chat-state' : ''}`}
      >
      {activeConversationId === null ? (
        <div className="welcome-screen">
          <h1>What&apos;s on your mind today?</h1>
        </div>
      ) : (
        <>
          <section
            ref={messagesAreaRef}
            className={`messages-area ${sending ? 'streaming-active' : ''}`}
            aria-live="polite"
            onScroll={(event) => {
              const element = event.currentTarget
              const distanceFromBottom =
                element.scrollHeight - element.scrollTop - element.clientHeight
              shouldFollowStreamRef.current = distanceFromBottom < 64
              setShowScrollToBottom((visible) =>
                distanceFromBottom > 180 ? true : visible && distanceFromBottom > 64,
              )
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
                        {item.role === 'user' &&
                          Boolean(item.attachments?.length) && (
                            <div className="message-image-attachments">
                              {item.attachments?.map((attachment) => (
                                <figure
                                  className="message-image-attachment"
                                  key={`${item.id}-${attachment.filename}`}
                                >
                                  <img
                                    src={attachment.data_url}
                                    alt={attachment.filename}
                                    loading="lazy"
                                  />
                                  <figcaption>{attachment.filename}</figcaption>
                                </figure>
                              ))}
                            </div>
                          )}
                        {item.role === 'assistant' ? (
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {item.content}
                          </ReactMarkdown>
                        ) : (
                          <p>{item.content}</p>
                        )}
                      </div>

                      {item.role === 'user' && (
                        <>
                          {copyErrorMessageId === item.id && (
                            <p className="copy-message-error" role="alert">
                              Could not copy this prompt. Try selecting and
                              copying the text.
                            </p>
                          )}
                          <div className="message-actions user-message-actions">
                            <button
                              type="button"
                              className={`message-action-button${copiedMessageId === item.id ? ' is-copied' : ''}`}
                              onClick={() =>
                                void handleCopyMessage(item.id, item.content)
                              }
                              aria-label={
                                copiedMessageId === item.id
                                  ? 'Prompt copied'
                                  : 'Copy prompt'
                              }
                              data-tooltip={
                                copiedMessageId === item.id
                                  ? 'Copied'
                                  : 'Copy'
                              }
                            >
                              {copiedMessageId === item.id ? (
                                <Check size={18} />
                              ) : (
                                <Copy size={18} />
                              )}
                            </button>
                            <button
                              type="button"
                              className="message-action-button"
                              onClick={() =>
                                void handleSharePrompt(item.content)
                              }
                              aria-label="Share prompt"
                              data-tooltip="Share"
                            >
                              <Share2 size={18} />
                            </button>
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
                              aria-label="Edit prompt"
                              data-tooltip="Edit"
                            >
                              <Pencil size={18} />
                            </button>
                          </div>
                        </>
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
                          <div className="message-actions assistant-message-actions">
                            <button
                              type="button"
                              className={`message-action-button${copiedMessageId === item.id ? ' is-copied' : ''}`}
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
                              className={`message-action-button${regeneratingMessageId === item.id ? ' is-regenerating' : ''}`}
                              onClick={() =>
                                void handleRegenerateAnswer(item.id)
                              }
                              disabled={
                                sending ||
                                loadingMessages ||
                                savingEditedMessage ||
                                regeneratingMessageId !== null
                              }
                              aria-label={
                                regeneratingMessageId === item.id
                                  ? 'Regenerating answer'
                                  : 'Regenerate answer'
                              }
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

          {showScrollToBottom && (
            <button
              type="button"
              className="scroll-to-bottom"
              aria-label="Scroll to latest message"
              title="Scroll to latest message"
              onClick={() => {
                const messagesArea = messagesAreaRef.current
                if (messagesArea) {
                  messagesArea.scrollTo({
                    top: messagesArea.scrollHeight,
                    behavior: 'smooth',
                  })
                }
              }}
            >
              <ArrowDown size={18} />
            </button>
          )}

        </>
      )}

      <div className="composer-container">
        {pendingAttachments.length > 0 && (
          <div className="composer-attachments" aria-label="Attached images">
            {pendingAttachments.map((attachment) => (
              <div className="composer-attachment" key={attachment.id}>
                <img
                  src={attachment.previewUrl}
                  alt={attachment.file.name}
                />
                <span title={attachment.file.name}>
                  {attachment.file.name}
                </span>
                <button
                  type="button"
                  className="composer-attachment-remove"
                  onClick={() => removePendingAttachment(attachment.id)}
                  aria-label={`Remove ${attachment.file.name}`}
                  disabled={sending}
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        {attachmentError && (
          <p className="attachment-error" role="alert">
            {attachmentError}
          </p>
        )}

        <form className="message-composer" onSubmit={handleSend}>
          <div className="composer-attachment-menu-wrap" ref={attachmentMenuRef}>
            <button
              type="button"
              className="composer-attach-button"
              onClick={() => setAttachmentMenuOpen((open) => !open)}
              aria-label="Add attachments"
              aria-haspopup="menu"
              aria-expanded={attachmentMenuOpen}
              title="Add photos and files"
              disabled={
                sending ||
                creatingChat ||
                pendingAttachments.length >= MAX_IMAGE_COUNT
              }
            >
              <Plus size={21} />
            </button>

            {attachmentMenuOpen && (
              <div className="composer-attachment-menu" role="menu">
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setAttachmentMenuOpen(false)
                    imageInputRef.current?.click()
                  }}
                >
                  <ImageIcon size={17} />
                  <span>Add photos &amp; files</span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setAttachmentMenuOpen(false)
                    cameraInputRef.current?.click()
                  }}
                >
                  <Camera size={17} />
                  <span>Take photo</span>
                </button>
              </div>
            )}
          </div>

          <textarea
            ref={composerInputRef}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Message AI Assistant..."
            rows={1}
            aria-label="Message"
          />

          <div className="composer-controls">
            <label className="model-picker">
              <span className="visually-hidden">Model</span>
              <select
                value={modelPreference}
                onChange={(event) => {
                  const preference = event.currentTarget.value
                  if (isModelPreference(preference)) {
                    setModelPreference(preference)
                  }
                }}
                aria-label="Choose model"
                disabled={sending}
              >
                <option value="auto">Auto</option>
                <option value="openai/gpt-oss-20b">GPT-OSS 20B</option>
                <option value="openai/gpt-oss-120b">GPT-OSS 120B</option>
              </select>
            </label>

            <button
              type="submit"
              className="send-button"
              disabled={
                (!message.trim() && pendingAttachments.length === 0) ||
                creatingChat ||
                sending ||
                loadingMessages ||
                regeneratingMessageId !== null
              }
              aria-label="Send message"
              title="Send message"
            >
              <Send size={17} />
            </button>
          </div>
        </form>

        <input
          ref={imageInputRef}
          className="visually-hidden"
          type="file"
          accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
          multiple
          onChange={(event) => {
            handleAttachmentSelection(event.currentTarget.files)
            event.currentTarget.value = ''
          }}
          aria-label="Choose image files"
        />
        <input
          ref={cameraInputRef}
          className="visually-hidden"
          type="file"
          accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
          capture="environment"
          onChange={(event) => {
            handleAttachmentSelection(event.currentTarget.files)
            event.currentTarget.value = ''
          }}
          aria-label="Take a photo"
        />

      </div>
      </div>
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
