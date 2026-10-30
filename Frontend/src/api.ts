
import axios from 'axios'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000/api/v1'
).replace(/\/+$/, '')

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use((config) => {
  const token = sessionStorage.getItem('access_token')

  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface User {
  id: number
  email: string
  created_at: string
}

export interface Conversation {
  id: number
  user_id: number
  title: string
  created_at: string
}

export interface ConversationCreate {
  user_id: number
  title: string
}

export interface ChatMessage {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface ChatResponse {
  user_message: ChatMessage
  assistant_message: ChatMessage
}

export async function login(email: string, password: string) {
  const response = await api.post<TokenResponse>('/auth/login', {
    email,
    password,
  })

  return response.data
}

export async function register(email: string, password: string) {
  const response = await api.post<User>('/auth/register', {
    email,
    password,
  })

  return response.data
}

export async function exchangeGoogleTicket(ticket: string) {
  const response = await api.post<TokenResponse>(
    '/auth/google/exchange',
    { ticket },
  )

  // Store the JWT so the existing Axios interceptor can use it.
  sessionStorage.setItem('access_token', response.data.access_token)

  return response.data
}

export async function createConversation(
  userId: number,
  title = 'New chat',
) {
  const response = await api.post<Conversation>(
    '/conversations',
    { user_id: userId, title },
  )

  return response.data
}

export async function renameConversation(
  conversationId: number,
  title: string,
) {
  const response = await api.patch<Conversation>(
    `/conversations/${conversationId}`,
    { title },
  )

  return response.data
}

export async function deleteConversation(conversationId: number) {
  await api.delete(`/conversations/${conversationId}`)
}

export async function getUserConversations(userId: number) {
  const response = await api.get<Conversation[]>(
    `/conversations/user/${userId}`,
  )

  return response.data
}


export async function searchConversations(query: string) {
  const response = await api.get<Conversation[]>('/conversations/search', {
    params: { q: query },
  })

  return response.data
}

export async function getConversationMessages(
  conversationId: number,
) {
  const response = await api.get<ChatMessage[]>(
    `/conversations/${conversationId}/messages`,
  )

  return response.data
}

export async function editMessage(
  conversationId: number,
  messageId: number,
  content: string,
) {
  const response = await api.put<ChatResponse>(
    `/conversations/${conversationId}/messages/${messageId}`,
    { content },
  )

  return response.data
}

export async function regenerateAssistantMessage(
  conversationId: number,
  messageId: number,
) {
  const response = await api.post<ChatMessage>(
    `/conversations/${conversationId}/messages/${messageId}/regenerate`,
  )

  return response.data
}

export async function sendChatMessage(
  conversationId: number,
  content: string,
) {
  const response = await api.post<ChatResponse>(
    `/conversations/${conversationId}/chat`,
    { content },
  )

  return response.data
}

async function consumeChatStream(
  path: string,
  body: object | undefined,
  onUserMessage: (message: ChatMessage) => void,
  onToken: (content: string) => void,
  signal: AbortSignal,
) {
  const token = sessionStorage.getItem('access_token')
  const headers: Record<string, string> = {
    Accept: 'text/event-stream',
    'Content-Type': 'application/json',
  }
  if (token) headers.Authorization = `Bearer ${token}`

  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      method: 'POST',
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    },
  )

  if (!response.ok) {
    const errorBody = await response.text()
    let detail = errorBody
    try {
      detail = JSON.parse(errorBody).detail || errorBody
    } catch {
      // Retain the response text when the server does not return JSON.
    }
    throw new Error(
      typeof detail === 'string'
        ? detail
        : 'Could not start the streaming response.',
    )
  }

  if (!response.body) {
    throw new Error('The server did not provide a response stream.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  const receivedMessages: {
    user: ChatMessage | null
    assistant: ChatMessage | null
  } = {
    user: null,
    assistant: null,
  }

  const processEvent = (event: string) => {
    const data = event
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
    if (!data) return

    const parsed: {
      type: string
      user_message?: ChatMessage
      assistant_message?: ChatMessage
      content?: string
      detail?: string
    } = JSON.parse(data)

    if (parsed.type === 'user' && parsed.user_message) {
      receivedMessages.user = parsed.user_message
      onUserMessage(parsed.user_message)
    } else if (parsed.type === 'token' && typeof parsed.content === 'string') {
      onToken(parsed.content)
    } else if (parsed.type === 'complete' && parsed.assistant_message) {
      receivedMessages.assistant = parsed.assistant_message
    } else if (parsed.type === 'error') {
      throw new Error(parsed.detail || 'The response stream failed.')
    }
  }

  try {
    while (true) {
      let result: ReadableStreamReadResult<Uint8Array>
      try {
        result = await reader.read()
      } catch (error) {
        if (receivedMessages.assistant) break
        throw error
      }

      const { done, value } = result
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      let boundary = buffer.match(/\r?\n\r?\n/)
      while (boundary && boundary.index !== undefined) {
        processEvent(buffer.slice(0, boundary.index))
        buffer = buffer.slice(boundary.index + boundary[0].length)
        boundary = buffer.match(/\r?\n\r?\n/)
      }
      if (receivedMessages.assistant) break
    }

    if (!receivedMessages.assistant) {
      buffer += decoder.decode()
      if (buffer.trim()) processEvent(buffer)
    }
  } finally {
    reader.releaseLock()
  }

  if (!receivedMessages.user || !receivedMessages.assistant) {
    throw new Error('The response stream ended before it was complete.')
  }

  return {
    user_message: receivedMessages.user,
    assistant_message: receivedMessages.assistant,
  }
}

export async function streamChatMessage(
  conversationId: number,
  content: string,
  onUserMessage: (message: ChatMessage) => void,
  onToken: (content: string) => void,
  signal: AbortSignal,
) {
  return consumeChatStream(
    `/conversations/${conversationId}/chat/stream`,
    { content },
    onUserMessage,
    onToken,
    signal,
  )
}

export async function retryChatMessageStream(
  conversationId: number,
  messageId: number,
  onUserMessage: (message: ChatMessage) => void,
  onToken: (content: string) => void,
  signal: AbortSignal,
) {
  return consumeChatStream(
    `/conversations/${conversationId}/messages/${messageId}/retry/stream`,
    undefined,
    onUserMessage,
    onToken,
    signal,
  )
}

export async function getCurrentUser() {
  const response = await api.get<User>('/auth/me')
  return response.data
}

export default api
