<script setup>
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'

const props = defineProps({ data: { type: Array, default: () => [] } })

const chartData = computed(() => {
  const depths = props.data.map(r => r.search_depth || 0)
  const max = Math.max(...depths, 1)
  const bins = new Array(max + 1).fill(0)
  depths.forEach(d => { if (d >= 0 && d <= max) bins[d]++ })
  return {
    labels: bins.map((_, i) => i),
    datasets: [{ label: 'Frequency', data: bins, backgroundColor: '#7c3aed' }]
  }
})
const options = { scales: { x: { title: { display: true, text: 'Depth' } }, y: { title: { display: true, text: 'Count' } } } }
</script>
<template><div class="chart-wrap"><Bar :data="chartData" :options="options" /></div></template>
<style scoped>.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }</style>
