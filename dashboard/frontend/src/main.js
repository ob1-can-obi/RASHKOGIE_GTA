import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router.js'
import { initChartDefaults } from './chartDefaults.js'
import './assets/main.css'

initChartDefaults()

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
