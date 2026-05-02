<script setup>
import { ref, onMounted } from 'vue'
import { useSessionsStore } from '../stores/sessions.js'
import StatusBadge from '../components/StatusBadge.vue'
import SessionComparisonChart from '../components/charts/SessionComparisonChart.vue'

const store = useSessionsStore()
const comparisonData = ref({})
const loading = ref(false)

onMounted(() => store.fetchSessions())

async function loadComparison() {
  if (store.comparisonIds.length < 2) return
  loading.value = true
  try {
    const ids = store.comparisonIds.join(',')
    const res = await fetch(`/api/sessions/compare/overlay?ids=${ids}`, {
      headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` }
    })
    if (res.ok) { const d = await res.json(); comparisonData.value = d.sessions || {} }
  } catch (e) { /* network */ }
  loading.value = false
}
</script>
<template>
  <div class="view">
    <h2>Sessions</h2>
    <div v-if="store.sessions.length === 0" class="empty">No past sessions. Complete a training run to see session history.</div>
    <template v-else>
      <table class="session-table">
        <thead><tr><th>Compare</th><th>Module</th><th>Session</th><th>Status</th><th>Started</th><th>Steps</th></tr></thead>
        <tbody>
          <tr v-for="s in store.sessions" :key="s.session_id">
            <td><input type="checkbox" :checked="store.comparisonIds.includes(s.session_id)" @change="store.toggleComparison(s.session_id)" /></td>
            <td>{{ s.module }}</td><td class="mono">{{ s.session_id?.slice(0, 8) }}</td>
            <td><StatusBadge :status="s.status" /></td><td>{{ s.started_at }}</td><td>{{ s.total_steps }}</td>
          </tr>
        </tbody>
      </table>
      <button class="btn-primary" :disabled="store.comparisonIds.length < 2" @click="loadComparison">Compare Selected</button>
      <SessionComparisonChart v-if="Object.keys(comparisonData).length" :sessions="comparisonData" />
    </template>
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.session-table { width: 100%; border-collapse: collapse; }
.session-table th, .session-table td { padding: var(--space-sm) var(--space-md); text-align: left; border-bottom: 1px solid var(--color-border); }
.session-table th { font-size: var(--font-size-label); color: var(--color-text-muted); }
.mono { font-family: monospace; }
.empty { color: var(--color-text-muted); text-align: center; padding: var(--space-xl); }
.btn-primary { background: var(--color-accent); color: white; align-self: flex-start; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
