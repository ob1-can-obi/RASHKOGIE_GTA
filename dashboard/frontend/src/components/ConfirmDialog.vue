<script setup>
defineProps({
  title: String,
  message: String,
  show: Boolean,
})
const emit = defineEmits(['confirm', 'cancel'])
</script>

<template>
  <Teleport to="body">
    <div v-if="show" class="dialog-overlay" @click.self="emit('cancel')">
      <div class="dialog-card">
        <h3>{{ title }}</h3>
        <p>{{ message }}</p>
        <div class="dialog-actions">
          <button class="btn-neutral" @click="emit('cancel')">Cancel</button>
          <button class="btn-destructive" @click="emit('confirm')">Confirm</button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.dialog-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 300;
}
.dialog-card {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius);
  padding: var(--space-xl);
  max-width: 400px;
  width: 90%;
}
.dialog-card h3 {
  font-size: var(--font-size-heading);
  margin-bottom: var(--space-sm);
}
.dialog-card p {
  color: var(--color-text-secondary);
  margin-bottom: var(--space-lg);
}
.dialog-actions {
  display: flex;
  gap: var(--space-md);
  justify-content: flex-end;
}
.btn-neutral {
  background: var(--color-bg-active);
  color: var(--color-text-primary);
}
.btn-destructive {
  background: var(--color-destructive);
  color: white;
}
</style>
