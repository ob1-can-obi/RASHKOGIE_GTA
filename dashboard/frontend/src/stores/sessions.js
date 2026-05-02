import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSessionsStore = defineStore('sessions', () => {
  const sessions = ref([])
  const comparisonIds = ref([])

  async function fetchSessions(module = null) {
    try {
      let url = '/api/sessions'
      if (module) url += `?module=${module}`
      const res = await fetch(url, {
        headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` }
      })
      if (res.ok) {
        const data = await res.json()
        sessions.value = data.sessions
      }
    } catch (e) { /* network error */ }
  }

  function toggleComparison(sessionId) {
    const idx = comparisonIds.value.indexOf(sessionId)
    if (idx >= 0) {
      comparisonIds.value.splice(idx, 1)
    } else {
      if (comparisonIds.value.length >= 5) comparisonIds.value.shift()
      comparisonIds.value.push(sessionId)
    }
  }

  return { sessions, comparisonIds, fetchSessions, toggleComparison }
})
