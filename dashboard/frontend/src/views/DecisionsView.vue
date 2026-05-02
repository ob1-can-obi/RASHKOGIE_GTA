<script setup>
import { useMetricsStore } from '../stores/metrics.js'
import DecisionHistogram from '../components/charts/DecisionHistogram.vue'
import NodesExpandedChart from '../components/charts/NodesExpandedChart.vue'
import SearchDepthChart from '../components/charts/SearchDepthChart.vue'

const store = useMetricsStore()
</script>
<template>
  <div class="view">
    <h2>Decisions</h2>
    <div v-if="store.decisions.length === 0" class="empty">No decision data. Start metacontroller training to see decision distributions.</div>
    <template v-else>
      <DecisionHistogram />
      <NodesExpandedChart />
      <SearchDepthChart :data="store.decisions" />
    </template>
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.empty { color: var(--color-text-muted); text-align: center; padding: var(--space-xl); }
</style>
