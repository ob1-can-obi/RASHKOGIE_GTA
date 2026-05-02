<script setup>
import { ref, onMounted } from 'vue'
import { useWebSocket } from '../composables/useWebSocket.js'
import ModuleSelector from '../components/ModuleSelector.vue'
import ConfirmDialog from '../components/ConfirmDialog.vue'

const { send } = useWebSocket()
const module = ref('encoder_intuition')
const checkpoints = ref([])
const showConfirm = ref(false)
const restoreTarget = ref(null)

async function fetchCheckpoints() {
  try {
    const res = await fetch(`/api/checkpoints?module=${module.value}`, { headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` } })
    if (res.ok) { const d = await res.json(); checkpoints.value = d.checkpoints || [] }
  } catch (e) { /* */ }
}

onMounted(fetchCheckpoints)

function askRestore(cp) { restoreTarget.value = cp; showConfirm.value = true }
function confirmRestore() {
  if (restoreTarget.value) {
    send({ type: 'restore_checkpoint', session_id: restoreTarget.value.session_id, module: module.value })
  }
  showConfirm.value = false; restoreTarget.value = null
}

function humanSize(bytes) {
  if (!bytes) return '--'
  const k = 1024; const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i]
}
</script>
<template>
  <div class="view">
    <div class="view-header"><h2>Weights</h2><ModuleSelector v-model="module" @update:modelValue="fetchCheckpoints" /></div>
    <div v-if="checkpoints.length === 0" class="empty">No checkpoints saved yet for this module.</div>
    <table v-else class="cp-table">
      <thead><tr><th>Session</th><th>Timestamp</th><th>Size</th><th>Actions</th></tr></thead>
      <tbody>
        <tr v-for="cp in checkpoints" :key="cp.filename">
          <td class="mono">{{ cp.session_id?.slice(0, 8) }}</td>
          <td>{{ cp.timestamp }}</td><td>{{ humanSize(cp.file_size) }}</td>
          <td class="actions">
            <a :href="`/api/checkpoints/${module}/${cp.session_id}/${cp.filename}`" download class="btn-sm">Download</a>
            <button class="btn-sm btn-warn" @click="askRestore(cp)">Restore</button>
          </td>
        </tr>
      </tbody>
    </table>
    <ConfirmDialog :show="showConfirm" title="Restore Checkpoint" message="This will overwrite the current model weights. Training progress since this checkpoint will be lost. Continue?" @confirm="confirmRestore" @cancel="showConfirm = false" />
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.view-header { display: flex; align-items: center; justify-content: space-between; }
.cp-table { width: 100%; border-collapse: collapse; }
.cp-table th, .cp-table td { padding: var(--space-sm) var(--space-md); text-align: left; border-bottom: 1px solid var(--color-border); }
.cp-table th { font-size: var(--font-size-label); color: var(--color-text-muted); }
.mono { font-family: monospace; }
.actions { display: flex; gap: var(--space-sm); }
.btn-sm { font-size: var(--font-size-label); padding: var(--space-xs) var(--space-sm); background: var(--color-bg-active); color: var(--color-text-primary); text-decoration: none; border-radius: var(--border-radius-sm); min-height: auto; }
.btn-warn { background: var(--color-warning); color: #000; }
.empty { color: var(--color-text-muted); text-align: center; padding: var(--space-xl); }
</style>
