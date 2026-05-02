<script setup>
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import { useMetricsStore } from '../../stores/metrics.js'

const store = useMetricsStore()
const chartData = computed(() => {
  const d = store.decisions
  const labels = d.map((_, i) => i)
  return {
    labels,
    datasets: [
      { label: 'EXPLORE', data: d.map(r => r.explore || 0), backgroundColor: '#3b82f6' },
      { label: 'ROLLBACK', data: d.map(r => r.rollback || 0), backgroundColor: '#ef4444' },
      { label: 'INTERRUPT', data: d.map(r => r.interrupt || 0), backgroundColor: '#f59e0b' },
      { label: 'COMMIT_NEXT', data: d.map(r => r.commit_next || 0), backgroundColor: '#10b981' },
    ]
  }
})
const options = { scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: 'Count' } } } }
</script>
<template><div class="chart-wrap"><Bar :data="chartData" :options="options" /></div></template>
<style scoped>.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }</style>
