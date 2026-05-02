# Dashboard Frontend

Vue 3 + Vite SPA for the training dashboard. Built output goes to `dist/` and is served by the FastAPI server.

## Development

```bash
cd dashboard/frontend
npm install
npm run dev        # dev server with hot reload
npm run build      # production build to dist/
```

## Structure

- `src/views/` — 7 page views (Metrics, Decisions, Hyperparams, Sessions, Embeddings, Predictions, Weights)
- `src/components/charts/` — 9 Chart.js chart components
- `src/stores/` — Pinia stores for metrics, params, and sessions state
- `src/composables/useWebSocket.js` — WebSocket client with 5s auto-reconnect
- `src/components/` — shared UI (Sidebar, AuthGate, MetricCard, StatusBadge, etc.)
