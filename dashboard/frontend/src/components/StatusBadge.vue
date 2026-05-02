<script setup>
defineProps({
  status: String,
})
</script>

<template>
  <span class="badge" :class="status?.toLowerCase()">{{ status }}</span>
</template>

<style scoped>
.badge {
  font-size: var(--font-size-label);
  padding: 2px var(--space-sm);
  border-radius: var(--border-radius-sm);
  font-weight: var(--font-weight-semibold);
}
.running { background: rgba(16, 185, 129, 0.15); color: var(--color-success); }
.converged { background: rgba(124, 58, 237, 0.15); color: var(--color-accent); }
.stopped { background: rgba(107, 114, 128, 0.15); color: var(--color-text-muted); }
</style>
