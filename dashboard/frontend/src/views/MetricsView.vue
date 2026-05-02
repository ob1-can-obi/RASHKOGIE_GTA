<script setup>
import { computed } from 'vue'
import { useMetricsStore } from '../stores/metrics.js'
import ModuleSelector from '../components/ModuleSelector.vue'
import MetricCard from '../components/MetricCard.vue'
import LossChart from '../components/charts/LossChart.vue'
import RewardChart from '../components/charts/RewardChart.vue'
import EpisodeReturnChart from '../components/charts/EpisodeReturnChart.vue'

const store = useMetricsStore()
const latestLoss = computed(() => { const m = store.currentMetrics; return m.length ? m[m.length - 1].loss?.toFixed(4) : '--' })
const latestReward = computed(() => { const m = store.currentMetrics; return m.length ? m[m.length - 1].reward?.toFixed(4) : '--' })
const latestReturn = computed(() => { const m = store.currentMetrics; return m.length ? m[m.length - 1].episode_return?.toFixed(4) : '--' })

function onModule(mod) { store.setModule(mod); store.fetchInitial(mod) }
</script>
<template>
  <div class="view">
    <div class="view-header"><h2>Metrics</h2><ModuleSelector :modelValue="store.selectedModule" @update:modelValue="onModule" /></div>
    <div v-if="store.currentMetrics.length === 0" class="empty">No training data yet. Start a training session to see live metrics.</div>
    <template v-else>
      <div class="cards"><MetricCard label="Current Loss" :value="latestLoss" /><MetricCard label="Current Reward" :value="latestReward" /><MetricCard label="Episode Return" :value="latestReturn" /></div>
      <LossChart /><RewardChart /><EpisodeReturnChart />
    </template>
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.view-header { display: flex; align-items: center; justify-content: space-between; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-md); }
.empty { color: var(--color-text-muted); text-align: center; padding: var(--space-xl); }
</style>
