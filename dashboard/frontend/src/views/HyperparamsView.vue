<script setup>
import { ref, watch } from 'vue'
import { useParamsStore } from '../stores/params.js'
import { useWebSocket } from '../composables/useWebSocket.js'
import ModuleSelector from '../components/ModuleSelector.vue'

const paramsStore = useParamsStore()
const { send } = useWebSocket()
const module = ref('encoder_intuition')
const form = ref({})
const errors = ref({})
const flashGreen = ref(false)

watch(module, (mod) => {
  form.value = { ...(paramsStore.config[mod] || {}) }
}, { immediate: true })

watch(() => paramsStore.config, () => {
  form.value = { ...(paramsStore.config[module.value] || {}) }
}, { deep: true })

function validate() {
  errors.value = {}
  if (form.value.lr != null && form.value.lr <= 0) errors.value.lr = 'Must be > 0'
  if (form.value.batch_size != null && (form.value.batch_size < 1 || !Number.isInteger(Number(form.value.batch_size)))) errors.value.batch_size = 'Must be integer >= 1'
  if (form.value.entropy_coeff != null && form.value.entropy_coeff < 0) errors.value.entropy_coeff = 'Must be >= 0'
  if (form.value.think_cost != null && form.value.think_cost < 0) errors.value.think_cost = 'Must be >= 0'
  return Object.keys(errors.value).length === 0
}

function apply() {
  if (!validate()) return
  send({ type: 'set_params', params: { module: module.value, ...form.value } })
  flashGreen.value = true
  setTimeout(() => { flashGreen.value = false }, 1000)
}

const fields = [
  { key: 'lr', label: 'Learning Rate', step: 0.0001 },
  { key: 'entropy_coeff', label: 'Entropy Coefficient', step: 0.001 },
  { key: 'think_cost', label: 'Think Cost', step: 0.001 },
  { key: 'batch_size', label: 'Batch Size', step: 1 },
  { key: 'convergence_threshold', label: 'Convergence Threshold', step: 0.01 },
  { key: 'convergence_patience', label: 'Convergence Patience', step: 1 },
]
</script>
<template>
  <div class="view">
    <div class="view-header"><h2>Hyperparameters</h2><ModuleSelector v-model="module" /></div>
    <div class="form-grid">
      <div v-for="f in fields" :key="f.key" class="field">
        <label>{{ f.label }}</label>
        <input type="number" v-model.number="form[f.key]" :step="f.step" :class="{ error: errors[f.key], flash: flashGreen }" />
        <span v-if="errors[f.key]" class="field-error">{{ errors[f.key] }}</span>
      </div>
    </div>
    <button class="btn-primary" :disabled="Object.keys(errors).length > 0" @click="apply">Apply Changes</button>
  </div>
</template>
<style scoped>
.view { display: flex; flex-direction: column; gap: var(--space-lg); }
.view-header { display: flex; align-items: center; justify-content: space-between; }
.form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--space-md); }
.field { display: flex; flex-direction: column; gap: var(--space-xs); }
.field label { font-size: var(--font-size-label); color: var(--color-text-secondary); }
.field input.error { border-color: var(--color-destructive); }
.field input.flash { border-color: var(--color-success); transition: border-color 0.3s; }
.field-error { font-size: var(--font-size-label); color: var(--color-destructive); }
.btn-primary { background: var(--color-accent); color: white; align-self: flex-start; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
