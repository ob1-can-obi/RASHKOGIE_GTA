<script setup>
import { onMounted, watch, ref } from 'vue'
import { useWebSocket } from './composables/useWebSocket.js'
import { useMetricsStore } from './stores/metrics.js'
import { useParamsStore } from './stores/params.js'
import Sidebar from './components/Sidebar.vue'
import AuthGate from './components/AuthGate.vue'
import ConnectionBanner from './components/ConnectionBanner.vue'

const { connected, messages, connect, send } = useWebSocket()
const metricsStore = useMetricsStore()
const paramsStore = useParamsStore()
const authenticated = ref(!sessionStorage.getItem('dashboard_requires_auth'))

function onAuthenticated() {
  authenticated.value = true
  connect()
  metricsStore.fetchInitial('encoder_intuition')
  paramsStore.fetchParams()
}

onMounted(async () => {
  try {
    const paramsRes = await fetch('/api/params')
    if (paramsRes.status === 401) {
      sessionStorage.setItem('dashboard_requires_auth', 'true')
      authenticated.value = false
    } else {
      authenticated.value = true
      connect()
      metricsStore.fetchInitial('encoder_intuition')
      paramsStore.fetchParams()
    }
  } catch (e) { /* server not reachable */ }
})

watch(messages, (msgs) => {
  while (msgs.length > 0) {
    const msg = msgs.shift()
    if (msg.type === 'metrics_update') {
      metricsStore.addMetrics(msg.module, msg.data || [])
    } else if (msg.type === 'decision_update') {
      metricsStore.addDecisions(msg.data || [])
    } else if (msg.type === 'param_ack') {
      paramsStore.updateFromAck(msg.params || {}, msg.config_version)
    }
  }
}, { deep: true })
</script>

<template>
  <AuthGate v-if="!authenticated" @authenticated="onAuthenticated" />
  <div v-else class="app-shell">
    <ConnectionBanner :connected="connected" />
    <Sidebar :connected="connected" />
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.app-shell { display: flex; min-height: 100vh; }
.main-content {
  flex: 1;
  padding: var(--space-xl);
  margin-left: var(--sidebar-width);
  overflow-y: auto;
}
@media (max-width: 1023px) { .main-content { margin-left: var(--sidebar-width-tablet); } }
@media (max-width: 767px) { .main-content { margin-left: 0; } }
</style>
