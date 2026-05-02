<script setup>
import { ref, watch, computed } from 'vue'
import { Line } from 'vue-chartjs'
import PredictionScatter from '../components/charts/PredictionScatter.vue'

const tab = ref('scatter')
const sessions = ref([])
const selectedSession = ref('')
const scatterPoints = ref([])
const errorData = ref([])

async function fetchSessions() {
  try {
    const res = await fetch('/api/embeddings/sessions', { headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` } })
    if (res.ok) { const d = await res.json(); sessions.value = d.sessions || [] }
  } catch (e) { /* */ }
}

async function fetchData() {
  if (!selectedSession.value) return
  try {
    if (tab.value === 'scatter') {
      const res = await fetch(`/api/predictions/scatter?session_id=${selectedSession.value}`, { headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` } })
      if (res.ok) { const d = await res.json(); scatterPoints.value = d.points || [] }
    } else {
      const res = await fetch(`/api/predictions/error?session_id=${selectedSession.value}`, { headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` } })
      if (res.ok) { const d = await res.json(); errorData.value = d.data || [] }
    }
  } catch (e) { /* */ }
}

fetchSessions()
watch([selectedSession, tab], fetchData)

const errorChartData = computed(() => ({
  labels: errorData.value.map(d => d.step),
  datasets: [{ label: 'MSE', data: errorData.value.map(d => d.mse), borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, fill: false }]
}))
</script>
<template>
  <div class="view">
    <h2>Predictions</h2>
    <div class="controls">
      <div class="tabs">
        <button :class="{ active: tab === 'scatter' }" @click="tab = 'scatter'">Predicted vs Actual</button>
        <button :class="{ active: tab === 'error' }" @click="tab = 'error'">Error Over Time</button>
      </div>
      <select v-model="selectedSession"><option value="">Select session...</option><option v-for="s in sessions" :key="s" :value="s">{{ s.slice(0, 8) }}</option></select>
    </div>
    <div v-if="!selectedSession" class="empty">Select a session to view prediction quality.</div>
    <PredictionScatter v-else-if="tab === 'scatter'" :points="scatterPoints" />
    <div v-else class="chart-wrap"><Line :data="errorChartData" :options="{ scales: { y: { title: { display: true, text: 'MSE' } } } }" /></div>
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.controls { display: flex; align-items: center; gap: var(--space-md); }
.tabs { display: flex; gap: var(--space-xs); }
.tabs button { background: var(--color-bg-secondary); color: var(--color-text-secondary); padding: var(--space-sm) var(--space-md); }
.tabs button.active { background: var(--color-accent); color: white; }
.empty { color: var(--color-text-muted); text-align: center; padding: var(--space-xl); }
.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }
</style>
