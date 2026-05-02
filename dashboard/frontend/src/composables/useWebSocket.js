import { ref, onUnmounted } from 'vue'

export function useWebSocket() {
  const connected = ref(false)
  const messages = ref([])
  let ws = null
  let reconnectTimer = null

  function getToken() {
    return sessionStorage.getItem('dashboard_token') || ''
  }

  function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const token = getToken()
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/browser?token=${encodeURIComponent(token)}`)

    ws.onopen = () => {
      connected.value = true
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        messages.value.push(data)
        if (messages.value.length > 1000) messages.value.splice(0, 500)
      } catch (e) { /* ignore non-JSON */ }
    }

    ws.onclose = () => {
      connected.value = false
      scheduleReconnect()
    }

    ws.onerror = () => {
      connected.value = false
    }
  }

  function scheduleReconnect() {
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        connect()
      }, 5000)
    }
  }

  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    }
  }

  function disconnect() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    if (ws) { ws.close(); ws = null }
  }

  onUnmounted(disconnect)

  return { connected, messages, connect, send, disconnect }
}
