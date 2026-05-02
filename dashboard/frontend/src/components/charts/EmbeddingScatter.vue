<script setup>
import { computed } from 'vue'
import { Scatter } from 'vue-chartjs'

const props = defineProps({
  points: { type: Array, default: () => [] },
  colorBy: { type: String, default: 'step' },
  explainedVariance: { type: Array, default: () => [] },
})

const contextColors = { straight: '#3b82f6', turn: '#f59e0b', braking: '#ef4444' }

const chartData = computed(() => {
  if (props.colorBy === 'context') {
    const groups = {}
    props.points.forEach(p => {
      const ctx = p.context || 'straight'
      if (!groups[ctx]) groups[ctx] = []
      groups[ctx].push({ x: p.x, y: p.y })
    })
    return {
      datasets: Object.entries(groups).map(([ctx, pts]) => ({
        label: ctx, data: pts, backgroundColor: contextColors[ctx] || '#7c3aed', pointRadius: 3,
      }))
    }
  }
  const maxStep = Math.max(...props.points.map(p => p.step || 0), 1)
  return {
    datasets: [{
      label: 'Embeddings',
      data: props.points.map(p => ({ x: p.x, y: p.y })),
      backgroundColor: props.points.map(p => {
        const t = (p.step || 0) / maxStep
        const r = Math.round(59 + t * 65), g = Math.round(130 - t * 72), b = Math.round(246 - t * 9)
        return `rgb(${r},${g},${b})`
      }),
      pointRadius: 3,
    }]
  }
})
const options = {
  scales: {
    x: { title: { display: true, text: 'PC1' } },
    y: { title: { display: true, text: 'PC2' } },
  }
}
</script>
<template>
  <div class="chart-wrap">
    <Scatter :data="chartData" :options="options" />
    <p v-if="explainedVariance.length" class="variance">PC1: {{ (explainedVariance[0] * 100).toFixed(1) }}%, PC2: {{ (explainedVariance[1] * 100).toFixed(1) }}%</p>
  </div>
</template>
<style scoped>
.chart-wrap { min-height: var(--chart-min-height); background: var(--color-bg-secondary); border-radius: var(--border-radius); padding: var(--space-md); }
.variance { font-size: var(--font-size-label); color: var(--color-text-muted); margin-top: var(--space-sm); }
</style>
