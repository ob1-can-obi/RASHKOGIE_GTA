<script setup>
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import { useMetricsStore } from '../../stores/metrics.js'

const store = useMetricsStore()
const chartData = computed(() => ({
  labels: store.steps,
  datasets: [{ label: 'Episode Return', data: store.episodeReturns, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, fill: false }]
}))
const options = { scales: { x: { title: { display: true, text: 'Step' } }, y: { title: { display: true, text: 'Episode Return' } } } }
</script>
<template><div class="chart-wrap"><Line :data="chartData" :options="options" /></div></template>
<style scoped>.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }</style>
