import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const MAX_DISPLAY_POINTS = 2000

export const useMetricsStore = defineStore('metrics', () => {
  const metricsByModule = ref({})
  const decisions = ref([])
  const selectedModule = ref('encoder_intuition')

  const currentMetrics = computed(() => metricsByModule.value[selectedModule.value] || [])
  const steps = computed(() => currentMetrics.value.map(r => r.step))
  const losses = computed(() => currentMetrics.value.map(r => r.loss))
  const rewards = computed(() => currentMetrics.value.map(r => r.reward))
  const episodeReturns = computed(() => currentMetrics.value.map(r => r.episode_return))

  function addMetrics(module, rows) {
    if (!metricsByModule.value[module]) metricsByModule.value[module] = []
    metricsByModule.value[module].push(...rows)
    if (metricsByModule.value[module].length > MAX_DISPLAY_POINTS) {
      metricsByModule.value[module] = metricsByModule.value[module].slice(-MAX_DISPLAY_POINTS)
    }
  }

  function addDecisions(rows) {
    decisions.value.push(...rows)
    if (decisions.value.length > MAX_DISPLAY_POINTS) {
      decisions.value = decisions.value.slice(-MAX_DISPLAY_POINTS)
    }
  }

  function setModule(module) { selectedModule.value = module }

  async function fetchInitial(module) {
    try {
      const res = await fetch(`/api/metrics/latest?module=${module}&limit=${MAX_DISPLAY_POINTS}`, {
        headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` }
      })
      if (res.ok) {
        const data = await res.json()
        metricsByModule.value[module] = data.rows
      }
    } catch (e) { /* network error */ }
  }

  return { metricsByModule, decisions, selectedModule, currentMetrics, steps, losses, rewards, episodeReturns, addMetrics, addDecisions, setModule, fetchInitial }
})
