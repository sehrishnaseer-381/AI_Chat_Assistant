
import axios from 'axios'

const api = axios.create({
  baseURL:
    import.meta.env.VITE_API_BASE_URL ||
    'http://127.0.0.1:8000/api/v1',
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

export async function getConversationMessages(
  conversationId: number,
) {
  const response = await api.get<ChatMessage[]>(
    `/conversations/${conversationId}/messages`,
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

export async function getCurrentUser() {
  const response = await api.get<User>('/auth/me')
  return response.data
}

export default api
