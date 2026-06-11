import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import type { RiskTimeline, RiskTimelinePayload, RiskTimelinePoint } from '../types/incident'

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
)

interface RiskTimelineChartProps {
  timeline: RiskTimeline
}

interface ChartPoint {
  x: number
  y: number
  frame_idx: number
  time_sec?: number
  shake_energy?: number
  jerk_energy?: number
  valid_motion?: boolean
}

function isTimelinePayload(timeline: RiskTimeline): timeline is RiskTimelinePayload {
  return !Array.isArray(timeline) && Array.isArray(timeline.points)
}

function formatNumber(value: number | undefined, digits = 2) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  })
}

function toChartPoints(timeline: RiskTimeline): ChartPoint[] {
  if (isTimelinePayload(timeline)) {
    return timeline.points
      .filter((p): p is RiskTimelinePoint => typeof p.risk_score === 'number')
      .map((p) => ({
        x: typeof p.time_sec === 'number' ? p.time_sec : p.frame_idx,
        y: p.risk_score,
        frame_idx: p.frame_idx,
        time_sec: p.time_sec,
        shake_energy: p.shake_energy,
        jerk_energy: p.jerk_energy,
        valid_motion: p.valid_motion,
      }))
  }

  return timeline.map((score, i) => ({
    x: i,
    y: score,
    frame_idx: i,
  }))
}

export default function RiskTimelineChart({ timeline }: RiskTimelineChartProps) {
  const payload = isTimelinePayload(timeline) ? timeline : null
  const points = toChartPoints(timeline)
  const xTitle = payload ? 'Time (seconds)' : 'Frame window'
  const peakPoint =
    payload?.peak && typeof payload.peak.risk_score === 'number'
      ? [
          {
            x:
              typeof payload.peak.time_sec === 'number'
                ? payload.peak.time_sec
                : payload.peak.frame_idx ?? 0,
            y: payload.peak.risk_score,
            frame_idx: payload.peak.frame_idx ?? 0,
            time_sec: payload.peak.time_sec,
          },
        ]
      : []

  return (
    <>
      {payload && (
        <div className="timeline-metrics">
          <div>
            <span>Peak risk</span>
            <strong>{formatNumber(payload.peak?.risk_score, 3)}</strong>
          </div>
          <div>
            <span>Peak time</span>
            <strong>{formatNumber(payload.peak?.time_sec)}s</strong>
          </div>
          <div>
            <span>Duration</span>
            <strong>{formatNumber(payload.duration_sec)}s</strong>
          </div>
          <div>
            <span>Frames</span>
            <strong>{formatNumber(payload.frame_count, 0)}</strong>
          </div>
          <div>
            <span>FPS</span>
            <strong>{formatNumber(payload.fps)}</strong>
          </div>
          <div>
            <span>Source</span>
            <strong>{payload.source ?? '—'}</strong>
          </div>
        </div>
      )}

      <div className="chart-container">
        <Line
          data={{
            datasets: [
              {
                label: 'Risk score',
                data: points,
                borderColor: 'rgb(219, 72, 72)',
                backgroundColor: 'rgba(219, 72, 72, 0.14)',
                fill: true,
                tension: 0.25,
                pointRadius: 0,
                parsing: false,
              },
              {
                label: 'Peak',
                data: peakPoint,
                borderColor: 'rgb(17, 24, 39)',
                backgroundColor: 'rgb(17, 24, 39)',
                pointRadius: 4,
                pointHoverRadius: 6,
                showLine: false,
                parsing: false,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: payload ? true : false },
              title: { display: false },
              tooltip: {
                callbacks: {
                  title: (items) => {
                    const raw = items[0]?.raw as ChartPoint | undefined
                    if (!raw) return ''
                    return `Frame ${raw.frame_idx}${raw.time_sec != null ? ` · ${formatNumber(raw.time_sec)}s` : ''}`
                  },
                  label: (item) => {
                    const raw = item.raw as ChartPoint | undefined
                    if (!raw) return ''
                    const parts = [`Risk ${formatNumber(raw.y, 3)}`]
                    if (raw.shake_energy != null) {
                      parts.push(`Shake ${formatNumber(raw.shake_energy, 2)}`)
                    }
                    if (raw.jerk_energy != null) {
                      parts.push(`Jerk ${formatNumber(raw.jerk_energy, 2)}`)
                    }
                    if (raw.valid_motion === false) {
                      parts.push('Invalid motion')
                    }
                    return parts.join(' · ')
                  },
                },
              },
            },
            scales: {
              x: {
                type: 'linear',
                title: { display: true, text: xTitle },
                ticks: { maxTicksLimit: 12 },
              },
              y: {
                min: 0,
                max: 1,
                title: { display: true, text: 'Relative motion risk' },
              },
            },
          }}
        />
      </div>
    </>
  )
}
