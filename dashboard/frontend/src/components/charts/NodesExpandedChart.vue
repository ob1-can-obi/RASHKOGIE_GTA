<script setup>
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import { useMetricsStore } from '../../stores/metrics.js'

const store = useMetricsStore()
const chartData = computed(() => ({
  labels: store.decisions.map((_, i) => i),
  datasets: [{ label: 'Nodes Expanded', data: store.decisions.map(r => r.nodes_expanded || 0), borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, fill: false }]
}))
const options = { scales: { y: { title: { display: true, text: 'Nodes' } } } }
</script>
<template><div class="chart-wrap"><Line :data="chartData" :options="options" /></div></template>
<style scoped>.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }</style>
