import React, { useMemo } from 'react'
import { AreaChart, Area, XAxis, YAxis, ReferenceLine, ResponsiveContainer } from 'recharts'
import { normalCDF } from '../utils/stats.js'

function pdf(x, mu, sigma) {
  if (sigma <= 0) return 0
  const z = (x - mu) / sigma
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI))
}

export default function NormalCurveChart({ prior, posterior, budget, color, showBoth = false, height = 110 }) {
  const dist = posterior || prior

  const data = useMemo(() => {
    if (!dist || dist.sigma <= 0) return []
    const all = showBoth && posterior ? [prior, posterior] : [dist]
    const lo = Math.min(...all.map(d => d.mu - 3.5 * d.sigma))
    const hi = Math.max(...all.map(d => d.mu + 3.5 * d.sigma))
    const n = 100, step = (hi - lo) / (n - 1)
    return Array.from({ length: n }, (_, i) => {
      const x = lo + i * step
      const pt = { x: Math.round(x) }
      if (showBoth && posterior) {
        pt.prior = pdf(x, prior.mu, prior.sigma)
        pt.post  = pdf(x, posterior.mu, posterior.sigma)
      } else {
        pt.y = pdf(x, dist.mu, dist.sigma)
      }
      return pt
    })
  }, [dist, prior, posterior, showBoth])

  if (!data.length) return null
  const gid = `cg${color.replace(/[^a-zA-Z0-9]/g, '')}`

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -32 }}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={color} stopOpacity={0.01}/>
          </linearGradient>
        </defs>
        <XAxis dataKey="x"
          tick={{ fill: '#3d4558', fontSize: 9, fontFamily: 'JetBrains Mono' }}
          tickFormatter={v => `$${v}`} tickLine={false} axisLine={false}
          interval="preserveStartEnd"/>
        <YAxis hide/>
        {budget != null && (
          <ReferenceLine x={budget} stroke={color} strokeWidth={1.5} strokeDasharray="4 3"/>
        )}
        {showBoth && posterior ? (
          <>
            <Area type="monotone" dataKey="prior" stroke="rgba(255,255,255,0.1)"
              strokeWidth={1} strokeDasharray="3 3" fill="none" dot={false}/>
            <Area type="monotone" dataKey="post" stroke={color}
              strokeWidth={2} fill={`url(#${gid})`} dot={false}/>
          </>
        ) : (
          <Area type="monotone" dataKey="y" stroke={color}
            strokeWidth={2} fill={`url(#${gid})`} dot={false}/>
        )}
      </AreaChart>
    </ResponsiveContainer>
  )
}
