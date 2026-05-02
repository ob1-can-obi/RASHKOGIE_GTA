<script setup>
defineProps({
  connected: Boolean,
})
</script>

<template>
  <div v-if="!connected" class="connection-banner">
    Connection lost. Reconnecting...
  </div>
</template>

<style scoped>
.connection-banner {
  position: fixed;
  top: 0;
  left: var(--sidebar-width);
  right: 0;
  background: var(--color-destructive);
  color: white;
  text-align: center;
  padding: var(--space-sm);
  font-size: var(--font-size-label);
  z-index: 200;
}
@media (max-width: 767px) {
  .connection-banner { left: 0; }
}
</style>
