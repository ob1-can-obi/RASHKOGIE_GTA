<script setup>
import { ref, watch } from 'vue'
import EmbeddingScatter from '../components/charts/EmbeddingScatter.vue'

const tab = ref('step')
const sessions = ref([])
const selectedSession = ref('')
const points = ref([])
const explainedVariance = ref([])

async function fetchSessions() {
  try {
    const res = await fetch('/api/embeddings/sessions', { headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` } })
    if (res.ok) { const d = await res.json(); sessions.value = d.sessions || [] }
  } catch (e) { /* */ }
}

async function fetchPoints() {
  if (!selectedSession.value) return
  try {
    const res = await fetch(`/api/embeddings/pca?session_id=${selectedSession.value}&color_by=${tab.value}`, { headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` } })
    if (res.ok) { const d = await res.json(); points.value = d.points || []; explainedVariance.value = d.explained_variance || [] }
  } catch (e) { /* */ }
}

fetchSessions()
watch([selectedSession, tab], fetchPoints)
</script>
<template>
  <div class="view">
    <h2>Embeddings</h2>
    <div class="controls">
      <div class="tabs">
        <button :class="{ active: tab === 'step' }" @click="tab = 'step'">Cluster Evolution</button>
        <button :class="{ active: tab === 'context' }" @click="tab = 'context'">State Types</button>
      </div>
      <select v-model="selectedSession"><option value="">Select session...</option><option v-for="s in sessions" :key="s" :value="s">{{ s.slice(0, 8) }}</option></select>
    </div>
    <div v-if="points.length === 0" class="empty">No embedding snapshots. Train with embedding logging enabled.</div>
    <EmbeddingScatter v-else :points="points" :colorBy="tab" :explainedVariance="explainedVariance" />
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.controls { display: flex; align-items: center; gap: var(--space-md); }
.tabs { display: flex; gap: var(--space-xs); }
.tabs button { background: var(--color-bg-secondary); color: var(--color-text-secondary); padding: var(--space-sm) var(--space-md); }
.tabs button.active { background: var(--color-accent); color: white; }
.empty { color: var(--color-text-muted); text-align: center; padding: var(--space-xl); }
</style>
