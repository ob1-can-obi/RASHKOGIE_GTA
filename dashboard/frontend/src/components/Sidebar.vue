<script setup>
defineProps({
  connected: Boolean
})
</script>

<template>
  <nav class="sidebar">
    <div class="sidebar-header">
      <h1 class="sidebar-title">RASHKOGIE</h1>
      <span class="sidebar-subtitle">Training Dashboard</span>
    </div>
    <ul class="sidebar-nav">
      <li><router-link to="/metrics">Metrics</router-link></li>
      <li><router-link to="/decisions">Decisions</router-link></li>
      <li><router-link to="/hyperparams">Hyperparams</router-link></li>
      <li><router-link to="/sessions">Sessions</router-link></li>
      <li><router-link to="/embeddings">Embeddings</router-link></li>
      <li><router-link to="/predictions">Predictions</router-link></li>
      <li><router-link to="/weights">Weights</router-link></li>
      <li><router-link to="/help">Help</router-link></li>
    </ul>
    <div class="sidebar-footer">
      <span class="status-dot" :class="connected ? 'live' : 'offline'"></span>
      <span class="status-text">{{ connected ? 'Live' : 'Disconnected' }}</span>
    </div>
  </nav>
</template>

<style scoped>
.sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: var(--sidebar-width);
  height: 100vh;
  background: var(--color-bg-secondary);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  z-index: 100;
}
.sidebar-header {
  padding: var(--space-lg) var(--space-md);
  border-bottom: 1px solid var(--color-border);
}
.sidebar-title {
  font-size: var(--font-size-heading);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-heading);
  color: var(--color-accent);
}
.sidebar-subtitle {
  font-size: var(--font-size-label);
  color: var(--color-text-muted);
}
.sidebar-nav {
  flex: 1;
  list-style: none;
  padding: var(--space-sm) 0;
}
.sidebar-nav li a {
  display: block;
  padding: var(--space-sm) var(--space-md);
  color: var(--color-text-secondary);
  text-decoration: none;
  border-left: 3px solid transparent;
  min-height: var(--touch-target-min);
  display: flex;
  align-items: center;
}
.sidebar-nav li a:hover {
  color: var(--color-text-primary);
  background: var(--color-bg-active);
}
.sidebar-nav li a.router-link-active {
  color: var(--color-text-primary);
  background: var(--color-bg-active);
  border-left-color: var(--color-accent);
}
.sidebar-footer {
  padding: var(--space-md);
  border-top: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.status-dot.live { background: var(--color-success); }
.status-dot.offline { background: var(--color-destructive); }
.status-text {
  font-size: var(--font-size-label);
  color: var(--color-text-muted);
}
@media (max-width: 1023px) {
  .sidebar { width: var(--sidebar-width-tablet); }
}
@media (max-width: 767px) {
  .sidebar { display: none; }
}
</style>
