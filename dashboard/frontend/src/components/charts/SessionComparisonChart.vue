<script setup>
import { computed } from 'vue'
import { Line } from 'vue-chartjs'

const props = defineProps({ sessions: { type: Object, default: () => ({}) } })
const overlayColors = ['#7c3aed', '#3b82f6', '#f59e0b', '#10b981', '#ec4899']

const chartData = computed(() => {
  const entries = Object.entries(props.sessions)
  const maxLen = Math.max(...entries.map(([, s]) => (s.metrics || []).length), 0)
  return {
    labels: Array.from({ length: maxLen }, (_, i) => i),
    datasets: entries.map(([id, s], i) => ({
      label: `${id} (${s.module || ''})`,
      data: (s.metrics || []).map(m => m.loss),
      borderColor: overlayColors[i % overlayColors.length],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
    }))
  }
})
const options = { scales: { x: { title: { display: true, text: 'Step' } }, y: { title: { display: true, text: 'Loss' } } } }
</script>
<template><div class="chart-wrap"><Line :data="chartData" :options="options" /></div></template>
<style scoped>.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }</style>
