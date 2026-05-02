import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/metrics' },
  { path: '/metrics', name: 'Metrics', component: () => import('./views/MetricsView.vue') },
  { path: '/decisions', name: 'Decisions', component: () => import('./views/DecisionsView.vue') },
  { path: '/hyperparams', name: 'Hyperparams', component: () => import('./views/HyperparamsView.vue') },
  { path: '/sessions', name: 'Sessions', component: () => import('./views/SessionsView.vue') },
  { path: '/embeddings', name: 'Embeddings', component: () => import('./views/EmbeddingsView.vue') },
  { path: '/predictions', name: 'Predictions', component: () => import('./views/PredictionsView.vue') },
  { path: '/weights', name: 'Weights', component: () => import('./views/WeightsView.vue') },
  { path: '/help', name: 'Help', component: () => import('./views/HelpView.vue') },
]

export default createRouter({ history: createWebHistory(), routes })
