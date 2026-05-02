<script setup>
import { ref } from 'vue'

const emit = defineEmits(['authenticated'])
const password = ref('')
const error = ref('')

async function submit() {
  error.value = ''
  sessionStorage.setItem('dashboard_token', password.value)
  try {
    const res = await fetch('/api/params', {
      headers: { 'Authorization': `Bearer ${password.value}` }
    })
    if (res.status === 401) {
      error.value = 'Invalid password. Please try again.'
      sessionStorage.removeItem('dashboard_token')
    } else {
      emit('authenticated')
    }
  } catch (e) {
    error.value = 'Cannot reach server.'
  }
}
</script>

<template>
  <div class="auth-gate">
    <div class="auth-card">
      <h2>RASHKOGIE Dashboard</h2>
      <p class="auth-hint">Enter the dashboard password to continue.</p>
      <form @submit.prevent="submit">
        <input type="password" v-model="password" placeholder="Password" autofocus />
        <button type="submit" class="btn-primary">Sign In</button>
      </form>
      <p v-if="error" class="auth-error">{{ error }}</p>
    </div>
  </div>
</template>

<style scoped>
.auth-gate {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: var(--color-bg-primary);
}
.auth-card {
  background: var(--color-bg-secondary);
  border-radius: var(--border-radius);
  padding: var(--space-xl);
  width: 360px;
  text-align: center;
}
.auth-card h2 {
  font-size: var(--font-size-heading);
  margin-bottom: var(--space-sm);
}
.auth-hint {
  color: var(--color-text-muted);
  font-size: var(--font-size-label);
  margin-bottom: var(--space-lg);
}
.auth-card form {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}
.auth-card input {
  width: 100%;
  font-family: inherit;
  font-size: var(--font-size-body);
  background: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius-sm);
  padding: var(--space-sm) var(--space-md);
  min-height: var(--touch-target-min);
}
.btn-primary {
  background: var(--color-accent);
  color: white;
}
.auth-error {
  color: var(--color-destructive);
  font-size: var(--font-size-label);
  margin-top: var(--space-md);
}
</style>
