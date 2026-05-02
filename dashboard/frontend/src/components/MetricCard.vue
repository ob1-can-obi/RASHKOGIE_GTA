<script setup>
defineProps({
  label: String,
  value: [String, Number],
  trend: String,
})
</script>

<template>
  <div class="metric-card">
    <span class="metric-label">{{ label }}</span>
    <span class="metric-value">{{ value ?? '--' }}</span>
    <span v-if="trend" class="metric-trend" :class="trend">{{ trend }}</span>
  </div>
</template>

<style scoped>
.metric-card {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius);
  padding: var(--space-md);
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
}
.metric-label {
  font-size: var(--font-size-label);
  color: var(--color-text-muted);
  line-height: var(--line-height-label);
}
.metric-value {
  font-size: var(--font-size-metric);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-metric);
}
.metric-trend {
  font-size: var(--font-size-label);
  color: var(--color-text-secondary);
}
</style>
