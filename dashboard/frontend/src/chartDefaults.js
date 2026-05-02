import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'

Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
)

export function initChartDefaults() {
  Chart.defaults.color = '#9ca3af'
  Chart.defaults.borderColor = '#2d3348'
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif'
  Chart.defaults.font.size = 12
  Chart.defaults.animation.duration = 0
  Chart.defaults.responsive = true
  Chart.defaults.maintainAspectRatio = false
}
