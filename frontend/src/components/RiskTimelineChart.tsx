import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
)

interface RiskTimelineChartProps {
  scores: number[]
}

export default function RiskTimelineChart({ scores }: RiskTimelineChartProps) {
  const labels = scores.map((_, i) => `${i + 1}`)

  return (
    <div className="chart-container">
      <Line
        data={{
          labels,
          datasets: [
            {
              label: 'Risk score',
              data: scores,
              borderColor: 'rgb(170, 59, 255)',
              backgroundColor: 'rgba(170, 59, 255, 0.15)',
              fill: true,
              tension: 0.3,
              pointRadius: 0,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: false },
          },
          scales: {
            x: {
              title: { display: true, text: 'Frame window' },
              ticks: { maxTicksLimit: 12 },
            },
            y: {
              min: 0,
              max: 1,
              title: { display: true, text: 'Risk' },
            },
          },
        }}
      />
    </div>
  )
}
