<script setup>
import { computed } from 'vue'
import { Scatter } from 'vue-chartjs'

const props = defineProps({ points: { type: Array, default: () => [] } })

const chartData = computed(() => {
  const pts = props.points
  const vals = pts.flatMap(p => [p.predicted, p.actual]).filter(v => v != null)
  const min = Math.min(...vals, 0)
  const max = Math.max(...vals, 1)
  return {
    datasets: [
      { label: 'Predictions', data: pts.map(p => ({ x: p.actual, y: p.predicted })), backgroundColor: '#7c3aed', pointRadius: 3 },
      { label: 'Ideal', data: [{ x: min, y: min }, { x: max, y: max }], borderColor: '#6b7280', borderDash: [5, 5], borderWidth: 1, pointRadius: 0, showLine: true, fill: false },
    ]
  }
})
const options = {
  scales: {
    x: { title: { display: true, text: 'Actual' } },
    y: { title: { display: true, text: 'Predicted' } },
  }
}
</script>
<template><div class="chart-wrap"><Scatter :data="chartData" :options="options" /></div></template>
<style scoped>.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }</style>
