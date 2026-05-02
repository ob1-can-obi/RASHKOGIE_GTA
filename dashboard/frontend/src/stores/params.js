import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useParamsStore = defineStore('params', () => {
  const config = ref({})
  const configVersion = ref(0)

  async function fetchParams() {
    try {
      const res = await fetch('/api/params', {
        headers: { 'Authorization': `Bearer ${sessionStorage.getItem('dashboard_token') || ''}` }
      })
      if (res.ok) {
        const data = await res.json()
        config.value = data.config
        configVersion.value = data.config_version
      }
    } catch (e) { /* network error */ }
  }

  function updateFromAck(params, version) {
    configVersion.value = version
    if (params.module && config.value[params.module]) {
      Object.assign(config.value[params.module], params)
    }
  }

  return { config, configVersion, fetchParams, updateFromAck }
})
