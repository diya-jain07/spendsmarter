import React, { useState, useMemo, useEffect } from 'react'
import { mleFitNormal, bayesianPosterior, sumOfNormals, probExceed, riskLevel } from './utils/stats.js'
import { CATEGORIES, DEMO_DATA } from './utils/categories.js'
import NormalCurveChart from './components/NormalCurveChart.jsx'

// ─── category config ────────────────────────────────────────────────────────
const CAT_COLOR = {
  groceries: '#5b6ef5', rent: '#9b7fff', utilities: '#f5a623',
  entertainment: '#00d4a0', misc: '#ff8c42',
}
const CAT_ICON  = { groceries:'🛒', rent:'🏠', utilities:'⚡', entertainment:'🎬', misc:'📦' }
const CAT_LABEL = { groceries:'Groceries', rent:'Rent', utilities:'Utilities', entertainment:'Entertainment', misc:'Miscellaneous' }

// ─── helpers ────────────────────────────────────────────────────────────────
function AnimNum({ to, dec = 1, suffix = '%' }) {
  const [v, setV] = useState(0)
  useEffect(() => {
    let cur = 0
    const inc = to / (700 / 14)
    const t = setInterval(() => {
      cur += inc
      if (cur >= to) { setV(to); clearInterval(t) } else setV(cur)
    }, 14)
    return () => clearInterval(t)
  }, [to])
  return <>{v.toFixed(dec)}{suffix}</>
}

function RiskTag({ label }) {
  const map = {
    High:   { c:'#ff4d6d', bg:'rgba(255,77,109,0.15)',  b:'rgba(255,77,109,0.3)'  },
    Medium: { c:'#f5a623', bg:'rgba(245,166,35,0.15)',  b:'rgba(245,166,35,0.3)'  },
    Low:    { c:'#00d4a0', bg:'rgba(0,212,160,0.15)',   b:'rgba(0,212,160,0.3)'   },
  }
  const s = map[label] || map.Medium
  return (
    <span style={{ fontSize:11, fontWeight:700, padding:'3px 10px', borderRadius:99,
      color:s.c, background:s.bg, border:`1px solid ${s.b}` }}>
      {label.toUpperCase()}
    </span>
  )
}

function NumInput({ value, onChange, width = 88 }) {
  const [focused, setFocused] = useState(false)
  return (
    <input type="number" min="0" placeholder="0" value={value}
      onChange={e => onChange(e.target.value)}
      onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
      style={{
        width, background: 'var(--input)',
        border: `1px solid ${focused ? 'rgba(255,255,255,0.25)' : 'var(--border2)'}`,
        borderRadius: 10, color: 'var(--text1)', fontFamily: 'var(--mono)',
        fontSize: 13, padding: '10px 12px', outline: 'none', transition: 'border-color 0.15s',
      }}
    />
  )
}

// ─── stat card (identical look to original site) ────────────────────────────
function StatCard({ tagLabel, tagColor, tagBg, value, dec = 1, suffix = '%', name, nameSub, color, barPct, dots }) {
  const iconKey = name?.toLowerCase().replace('miscellaneous','misc').replace(/\s/g,'')
  return (
    <div style={{ background:'var(--card)', border:'1px solid var(--border)',
      borderRadius:16, padding:'22px 24px' }}>
      <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:16 }}>
        <div style={{ width:40, height:40, borderRadius:10, background:`${color}22`,
          border:`1px solid ${color}44`, display:'flex', alignItems:'center',
          justifyContent:'center', fontSize:18 }}>
          {CAT_ICON[iconKey] || '📊'}
        </div>
        <span style={{ fontSize:11, fontWeight:700, padding:'3px 10px', borderRadius:99,
          color:tagColor, background:tagBg, border:`1px solid ${tagColor}44` }}>
          {tagLabel}
        </span>
      </div>
      <div style={{ fontSize:44, fontWeight:900, color, lineHeight:1, marginBottom:8,
        animation: value !== null ? 'numPop 0.4s ease both' : 'none' }}>
        {value !== null ? <AnimNum to={value} dec={dec} suffix={suffix}/> : '—'}
      </div>
      <div style={{ fontSize:15, fontWeight:700, color:'var(--text1)', marginBottom:4 }}>
        {name || '—'}
      </div>
      <div style={{ fontSize:12, color:'var(--text2)', marginBottom:14 }}>
        {nameSub || ''}
      </div>
      {dots && (
        <div style={{ display:'flex', gap:6, marginBottom:12 }}>
          {dots.map((c,i) => (
            <div key={i} style={{ width:10, height:10, borderRadius:'50%',
              background:c, boxShadow:`0 0 6px ${c}88` }}/>
          ))}
        </div>
      )}
      <div style={{ height:3, borderRadius:99, background:'rgba(255,255,255,0.06)', overflow:'hidden' }}>
        <div style={{ height:'100%', borderRadius:99, background:color,
          width: `${barPct ?? 0}%`, transition:'width 0.9s ease',
          boxShadow:`0 0 8px ${color}66` }}/>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════
export default function App() {
  const [history,  setHistory]  = useState(Object.fromEntries(CATEGORIES.map(c => [c.key, ['','','']])))
  const [budget,   setBudget]   = useState(Object.fromEntries(CATEGORIES.map(c => [c.key, ''])))
  const [midMonth, setMidMonth] = useState(Object.fromEntries(CATEGORIES.map(c => [c.key, ''])))
  const [tFrac,    setTFrac]    = useState(0.5)
  const [barW,     setBarW]     = useState({})

  function loadDemo() {
    setHistory(DEMO_DATA.history)
    setBudget(DEMO_DATA.budget)
    setMidMonth(DEMO_DATA.midMonth)
    setTFrac(DEMO_DATA.tFraction)
  }

  // ── priors: MLE fit from history ──────────────────────────────────────
  const priors = useMemo(() => {
    const r = {}
    CATEGORIES.forEach(cat => {
      const s = (history[cat.key] || []).map(Number).filter(v => !isNaN(v) && v > 0)
      r[cat.key] = s.length >= 2
        ? mleFitNormal(s)
        : { mu: cat.defaultMu, sigma: cat.defaultSigma }
    })
    return r
  }, [history])

  // ── posteriors: Bayesian update from mid-month spend ──────────────────
  const posteriors = useMemo(() => {
    const r = {}
    CATEGORIES.forEach(cat => {
      const s = Number(midMonth[cat.key])
      const t = Number(tFrac)
      const prior = priors[cat.key]
      r[cat.key] = (!isNaN(s) && s > 0 && t > 0)
        ? bayesianPosterior(prior.mu, prior.sigma, s, t)
        : null
    })
    return r
  }, [priors, midMonth, tFrac])

  // ── active dist: posterior if available, else prior ───────────────────
  const activeDists = useMemo(() => {
    const r = {}
    CATEGORIES.forEach(cat => { r[cat.key] = posteriors[cat.key] || priors[cat.key] })
    return r
  }, [priors, posteriors])

  // ── per-category probabilities ────────────────────────────────────────
  const catProbs = useMemo(() => {
    const r = {}
    CATEGORIES.forEach(cat => {
      const b = Number(budget[cat.key])
      const d = activeDists[cat.key]
      r[cat.key] = b > 0 ? probExceed(b, d.mu, d.sigma) : 0
    })
    return r
  }, [activeDists, budget])

  // ── total ─────────────────────────────────────────────────────────────
  const totalDist     = useMemo(() => sumOfNormals(CATEGORIES.map(c => activeDists[c.key])), [activeDists])
  const totalBudget   = CATEGORIES.reduce((a, cat) => a + (Number(budget[cat.key]) || 0), 0)
  const totalProbOver = totalBudget > 0 ? probExceed(totalBudget, totalDist.mu, totalDist.sigma) : null

  // ── risk summary derived values ───────────────────────────────────────
  const hasBudgets = CATEGORIES.some(c => Number(budget[c.key]) > 0)

  const highestKey = hasBudgets
    ? CATEGORIES.reduce((a, c) => catProbs[c.key] > catProbs[a] ? c.key : a, CATEGORIES[0].key)
    : null
  const safestKey = hasBudgets
    ? CATEGORIES.reduce((a, c) => catProbs[c.key] < catProbs[a] ? c.key : a, CATEGORIES[0].key)
    : null
  const highRiskCount = hasBudgets
    ? CATEGORIES.filter(c => catProbs[c.key] >= 0.65).length
    : null

  // animate bars when probs change
  useEffect(() => {
    if (hasBudgets) {
      setBarW({})
      setTimeout(() => {
        const w = {}
        CATEGORIES.forEach(c => { w[c.key] = catProbs[c.key] * 100 })
        setBarW(w)
      }, 80)
    }
  }, [catProbs, hasBudgets])

  // ─────────────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── HEADER ─────────────────────────────────────────────────────── */}
      <header style={{ background: 'rgba(13,15,20,0.95)', borderBottom: '1px solid var(--border)',
        backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 50 }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 28px', height: 58,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 34, height: 34, borderRadius: 9, background: 'rgba(91,110,245,0.2)',
              border: '1px solid rgba(91,110,245,0.35)', display: 'flex', alignItems: 'center',
              justifyContent: 'center', fontSize: 16 }}>✓</div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700 }}>Spend Smarter</div>
              <div style={{ fontSize: 11, color: 'var(--text2)' }}>Budget Risk Predictor</div>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#00d4a0',
            background: 'rgba(0,212,160,0.1)', border: '1px solid rgba(0,212,160,0.25)',
            padding: '6px 14px', borderRadius: 99 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#00d4a0',
              boxShadow: '0 0 6px #00d4a0' }}/>
            Model ready
          </div>
        </div>
      </header>

      <main style={{ flex: 1, maxWidth: 1100, margin: '0 auto', width: '100%', padding: '0 28px 56px' }}>

        {/* ── HERO ─────────────────────────────────────────────────────── */}
        <div style={{ textAlign: 'center', padding: '52px 24px 40px' }}>
          <h1 style={{ fontSize: 44, fontWeight: 900, letterSpacing: '-0.02em', lineHeight: 1.1, marginBottom: 14 }}>
            How is this month going?
          </h1>
          <p style={{ fontSize: 16, color: 'var(--text2)', lineHeight: 1.6, maxWidth: 480, margin: '0 auto' }}>
            Enter your spending history and how much you've spent so far this month, and see probability updates in real time as uncertainty shrinks!
          </p>
        </div>

        {/* ── MONTH PROGRESS SLIDER ────────────────────────────────────── */}
        <div style={{ background: 'var(--card)', border: '1px solid var(--border)',
          borderRadius: 16, padding: '22px 24px', marginBottom: 16,
          display: 'grid', gridTemplateColumns: '1fr auto', gap: 24, alignItems: 'center' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
              <span style={{ fontSize: 14, fontWeight: 600 }}>Month Progress</span>
              <span style={{ fontSize: 14, fontFamily: 'var(--mono)', color: 'var(--text1)' }}>
                Day {Math.round(tFrac * 30)} / 30
              </span>
            </div>
            <input type="range" min="0.01" max="0.99" step="0.01"
              value={tFrac} onChange={e => setTFrac(parseFloat(e.target.value))}/>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6,
              fontSize: 11, color: 'var(--text3)' }}>
              <span>Start of month</span><span>Mid</span><span>End of month</span>
            </div>
          </div>
          <button onClick={loadDemo} style={{ background: 'var(--raised)',
            border: '1px solid var(--border2)', borderRadius: 10, padding: '10px 20px',
            color: 'var(--text1)', fontSize: 13, fontWeight: 500,
            cursor: 'pointer', fontFamily: 'var(--font)', whiteSpace: 'nowrap' }}>
            Load demo data
          </button>
        </div>

        {/* ── CATEGORY INPUT CARDS ─────────────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 40 }}>
          {CATEGORIES.map(cat => {
            const prior = priors[cat.key]
            const post  = posteriors[cat.key]
            const b     = Number(budget[cat.key]) || null
            const activeProb = b ? catProbs[cat.key] : null
            const r = activeProb !== null ? riskLevel(activeProb) : null
            const varReduction = post ? Math.round((1 - post.sigma / prior.sigma) * 100) : null

            return (
              <div key={cat.key} style={{ background: 'var(--card)',
                border: `1px solid ${CAT_COLOR[cat.key]}22`, borderRadius: 16, padding: '22px 24px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>

                  {/* Left: inputs */}
                  <div>
                    {/* Title row */}
                    <div style={{ display: 'flex', alignItems: 'center',
                      justifyContent: 'space-between', marginBottom: 18 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div style={{ width: 36, height: 36, borderRadius: 9,
                          background: `${CAT_COLOR[cat.key]}22`, border: `1px solid ${CAT_COLOR[cat.key]}44`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>
                          {CAT_ICON[cat.key]}
                        </div>
                        <span style={{ fontSize: 15, fontWeight: 700, color: CAT_COLOR[cat.key] }}>
                          {cat.label}
                        </span>
                      </div>
                      {r && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <RiskTag label={r.label}/>
                          <span style={{ fontSize: 20, fontWeight: 800,
                            color: r.color, fontFamily: 'var(--mono)' }}>
                            {(activeProb * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Input fields row */}
                    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 14 }}>
                      {/* Past months */}
                      <div>
                        <div style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 7 }}>Past months ($)</div>
                        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                          {(history[cat.key] || []).map((v, idx) => (
                            <NumInput key={idx} value={v} width={72}
                              onChange={val => setHistory(h => ({
                                ...h, [cat.key]: h[cat.key].map((x, i) => i === idx ? val : x)
                              }))}/>
                          ))}
                          {(history[cat.key] || []).length < 6 && (
                            <button onClick={() => setHistory(h => ({
                              ...h, [cat.key]: [...h[cat.key], '']
                            }))} style={{ width: 72, background: 'none',
                              border: '1px dashed var(--border2)', borderRadius: 10,
                              color: 'var(--text3)', fontSize: 11,
                              cursor: 'pointer', padding: '10px 0' }}>
                              + mo
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Budget + spent so far */}
                      <div style={{ display: 'flex', gap: 10 }}>
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 7 }}>Budget ($)</div>
                          <NumInput value={budget[cat.key] || ''} width={90}
                            onChange={v => setBudget(b => ({ ...b, [cat.key]: v }))}/>
                        </div>
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 7 }}>Spent so far ($)</div>
                          <NumInput value={midMonth[cat.key]} width={100}
                            onChange={v => setMidMonth(m => ({ ...m, [cat.key]: v }))}/>
                        </div>
                      </div>
                    </div>

                    {/* Posterior σ reduction pill */}
                    {post && varReduction !== null && (
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10,
                        background: 'var(--raised)', borderRadius: 10, padding: '9px 14px' }}>
                        <div>
                          <div style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 1 }}>Prior σ</div>
                          <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--mono)' }}>
                            ${prior.sigma.toFixed(0)}
                          </div>
                        </div>
                        <span style={{ color: 'var(--text3)' }}>→</span>
                        <div>
                          <div style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 1 }}>Posterior σ</div>
                          <div style={{ fontSize: 16, fontWeight: 700,
                            fontFamily: 'var(--mono)', color: 'var(--green)' }}>
                            ${post.sigma.toFixed(0)}
                          </div>
                        </div>
                        <div style={{ background: 'rgba(0,212,160,0.12)',
                          border: '1px solid rgba(0,212,160,0.3)', borderRadius: 99,
                          padding: '4px 11px', fontSize: 12,
                          color: 'var(--green)', fontFamily: 'var(--mono)' }}>
                          ↓{varReduction}% uncertainty
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Right: live curve */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {activeProb !== null && (
                      <div style={{ fontSize: 26, fontWeight: 800, color: r.color, lineHeight: 1,
                        animation: 'numPop 0.4s ease both' }}>
                        {(activeProb * 100).toFixed(1)}%
                        <span style={{ fontSize: 12, color: 'var(--text2)',
                          fontWeight: 400, marginLeft: 8 }}>overspend probability</span>
                      </div>
                    )}
                    <NormalCurveChart
                      prior={prior} posterior={post} budget={b}
                      color={CAT_COLOR[cat.key]} height={120} showBoth={!!post}/>
                    <div style={{ display: 'flex', gap: 12 }}>
                      <span style={{ fontSize: 10, color: 'var(--text2)', fontFamily: 'var(--mono)' }}>
                        μ ${(post || prior).mu.toFixed(0)}
                      </span>
                      <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
                        σ ${(post || prior).sigma.toFixed(0)}
                      </span>
                      {post && (
                        <span style={{ fontSize: 10, color: 'var(--green)', fontFamily: 'var(--mono)' }}>
                          posterior updated
                        </span>
                      )}
                    </div>
                  </div>

                </div>
              </div>
            )
          })}
        </div>

        {/* ── YOUR RISK SUMMARY ────────────────────────────────────────── */}
        {hasBudgets && (
          <div style={{ animation: 'fadeUp 0.4s ease' }}>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>Your Risk Summary</div>
              <div style={{ fontSize: 13, color: 'var(--text2)' }}>
                Derived from your Bayesian-updated distributions
              </div>
            </div>

            {/* Stat cards — identical look to original site */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, marginBottom: 20 }}>
              <StatCard
                tagLabel="⚠ Highest Risk" tagColor="#ff4d6d" tagBg="rgba(255,77,109,0.15)"
                value={highestKey ? catProbs[highestKey] * 100 : null}
                name={highestKey ? CAT_LABEL[highestKey] : null}
                nameSub="probability of overspending"
                color="#ff4d6d"
                barPct={highestKey ? catProbs[highestKey] * 100 : 0}
              />
              <StatCard
                tagLabel="✓ Safest" tagColor="#00d4a0" tagBg="rgba(0,212,160,0.15)"
                value={safestKey ? catProbs[safestKey] * 100 : null}
                name={safestKey ? CAT_LABEL[safestKey] : null}
                nameSub="lowest overspend risk"
                color="#00d4a0"
                barPct={safestKey ? catProbs[safestKey] * 100 : 0}
              />
              <StatCard
                tagLabel="High Risk Count" tagColor="#f5a623" tagBg="rgba(245,166,35,0.15)"
                value={highRiskCount} dec={0} suffix="/5"
                name={highRiskCount === 0 ? 'All categories safe' : `${highRiskCount} categor${highRiskCount === 1 ? 'y' : 'ies'}`}
                nameSub="categories above 65% threshold"
                color="#f5a623"
                barPct={highRiskCount !== null ? (highRiskCount / 5) * 100 : 0}
                dots={CATEGORIES.map(c => riskLevel(catProbs[c.key]).color)}
              />
            </div>

            {/* Category risk breakdown — same as original */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

              {/* Risk bars */}
              <div style={{ background: 'var(--card)', border: '1px solid var(--border)',
                borderRadius: 16, padding: '24px' }}>
                <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>
                  Category Risk Breakdown
                </div>
                <div style={{ fontSize: 13, color: 'var(--text2)', marginBottom: 20 }}>
                  Probability that you overspend in each category
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  {CATEGORIES.map((cat, i) => {
                    const prob = catProbs[cat.key]
                    const r = riskLevel(prob)
                    const b = Number(budget[cat.key]) || 0
                    const s = midMonth[cat.key] ? Number(midMonth[cat.key]) : null
                    const totalB = CATEGORIES.reduce((a,c) => a+(Number(budget[c.key])||0), 0)
                    const share = totalB > 0 ? ((b/totalB)*100).toFixed(0) : 0
                    return (
                      <div key={cat.key}>
                        <div style={{ display: 'flex', justifyContent: 'space-between',
                          alignItems: 'center', marginBottom: 8 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <div style={{ width: 34, height: 34, borderRadius: 9,
                              background: `${CAT_COLOR[cat.key]}22`,
                              border: `1px solid ${CAT_COLOR[cat.key]}44`,
                              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16 }}>
                              {CAT_ICON[cat.key]}
                            </div>
                            <div>
                              <div style={{ fontSize: 13, fontWeight: 600 }}>{cat.label}</div>
                              <div style={{ fontSize: 11, color: 'var(--text2)' }}>
                                {s ? `$${s} spent · ` : ''}{b ? `$${b} budget` : 'no budget set'}{share ? ` · ${share}% of total` : ''}
                              </div>
                            </div>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <RiskTag label={r.label}/>
                            <span style={{ fontSize: 20, fontWeight: 800, color: r.color,
                              fontFamily: 'var(--mono)', minWidth: 46, textAlign: 'right' }}>
                              {Math.round(prob * 100)}%
                            </span>
                          </div>
                        </div>
                        <div style={{ height: 5, background: 'rgba(255,255,255,0.05)',
                          borderRadius: 99, overflow: 'hidden' }}>
                          <div style={{
                            height: '100%', borderRadius: 99,
                            width: `${barW[cat.key] || 0}%`,
                            background: `linear-gradient(90deg, ${r.color}88, ${r.color})`,
                            boxShadow: `0 0 8px ${r.color}55`,
                            transition: `width 0.75s cubic-bezier(0.4,0,0.2,1) ${i * 70}ms`,
                          }}/>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Total budget risk card */}
              <div style={{ background: 'var(--card)', border: '1px solid rgba(91,110,245,0.25)',
                borderRadius: 16, padding: '24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>Total Budget Risk</div>
                  <div style={{ fontSize: 13, color: 'var(--text2)' }}>
                    
                  </div>
                </div>
                {totalProbOver !== null ? (
                  <>
                    <div style={{ fontSize: 52, fontWeight: 900,
                      color: riskLevel(totalProbOver).color, lineHeight: 1,
                      animation: 'numPop 0.4s ease both' }}>
                      <AnimNum to={totalProbOver * 100}/>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--mono)' }}>
                      μ ${totalDist.mu.toFixed(0)} · σ ${totalDist.sigma.toFixed(0)} · budget ${totalBudget}
                    </div>
                    <NormalCurveChart
                      prior={totalDist} posterior={null} budget={totalBudget}
                      color={riskLevel(totalProbOver).color} height={130}/>
                  </>
                ) : (
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: 'var(--text3)', fontSize: 14 }}>
                    Set budgets above to see total risk
                  </div>
                )}
              </div>

            </div>
          </div>
        )}
      </main>

      <footer style={{ borderTop: '1px solid var(--border)', padding: '14px 28px', textAlign: 'center' }}>
        <span style={{ fontSize: 11, color: 'var(--text3)' }}>
          
        </span>
      </footer>
    </div>
  )
}
