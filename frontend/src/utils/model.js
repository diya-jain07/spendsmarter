/**
 * model.js — Logistic regression classifier for overspend risk.
 *
 * CS109 concepts:
 *   - Each output is Bernoulli(p_k) — independent per category
 *   - p_k estimated via MLE on Bernoulli likelihood
 *   - Logistic regression: p_k = σ(βᵀx)
 *   - Trained via gradient ascent on log-likelihood (= minimizing BCE)
 *   - BCE loss = cross-entropy = KL divergence minimization
 *
 * Implemented in pure JS — no ML libraries.
 */

export const CATEGORIES = ['groceries', 'rent', 'utilities', 'entertainment', 'misc']

// ─── Sigmoid ───────────────────────────────────────────────────────────────
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

// ─── Synthetic training data ───────────────────────────────────────────────
// Simulates 10,000 users with category-specific behavioral distributions.
// Spend ratio = actual / budget. Ratio > 1.0 → overspent → label = 1.
// Normal distributions motivated by CLT (monthly spend = sum of many purchases).
const BEHAVIORAL_PROFILES = [
  { muRatio: 1.05, sigmaRatio: 0.20 }, // groceries
  { muRatio: 0.98, sigmaRatio: 0.05 }, // rent
  { muRatio: 1.00, sigmaRatio: 0.25 }, // utilities
  { muRatio: 1.20, sigmaRatio: 0.35 }, // entertainment — highest overspend tendency
  { muRatio: 1.08, sigmaRatio: 0.30 }, // misc
]

const BUDGET_PROFILES = [
  { mu: 400,  sigma: 80  },
  { mu: 1400, sigma: 200 },
  { mu: 130,  sigma: 30  },
  { mu: 250,  sigma: 100 },
  { mu: 150,  sigma: 60  },
]

function randn() {
  // Box-Muller transform — standard Normal sample
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

function generateTrainingData(nSamples = 5000, seed = 42) {
  // Seeded-ish RNG for reproducibility
  const X = [], y = []
  for (let n = 0; n < nSamples; n++) {
    const budgets = BUDGET_PROFILES.map(p => Math.max(50, p.mu + p.sigma * randn()))
    const ratios  = BEHAVIORAL_PROFILES.map(p => Math.max(0.1, p.muRatio + p.sigmaRatio * randn()))
    const spend   = budgets.map((b, k) => b * ratios[k])
    const labels  = spend.map((s, k) => s > budgets[k] ? 1 : 0)
    // Feature vector: [spend×5, budget×5], normalized by dividing by 1000
    const features = [...spend.map(v => v / 1000), ...budgets.map(v => v / 1000)]
    X.push(features)
    y.push(labels)
  }
  return { X, y }
}

// ─── Logistic Regression — MLE via gradient ascent ─────────────────────────
// For each category k, we fit: p_k = σ(βₖᵀx)
// Log-likelihood: log L(β) = Σ_n [y_n log σ(βᵀxₙ) + (1-y_n) log(1-σ(βᵀxₙ))]
// Gradient: ∂log L/∂β = Σ_n xₙ(yₙ - σ(βᵀxₙ))   ← clean BCE+sigmoid gradient
//
// We train 5 independent logistic regressors (one per category).
function trainLogisticRegression(X, y, lr = 0.05, epochs = 80) {
  const nFeatures = X[0].length
  const nCats = y[0].length
  // Initialize weights near zero — standard practice
  const beta = Array.from({ length: nCats }, () => new Array(nFeatures).fill(0))
  const bias  = new Array(nCats).fill(0)

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradBeta = Array.from({ length: nCats }, () => new Array(nFeatures).fill(0))
    const gradBias = new Array(nCats).fill(0)

    for (let n = 0; n < X.length; n++) {
      const x = X[n]
      for (let k = 0; k < nCats; k++) {
        // Forward pass: σ(βᵀx + b)
        const logit = bias[k] + beta[k].reduce((s, w, j) => s + w * x[j], 0)
        const pred  = sigmoid(logit)
        // Gradient of log-likelihood: xₙ(yₙ - ŷₙ)
        const err   = y[n][k] - pred
        for (let j = 0; j < nFeatures; j++) gradBeta[k][j] += x[j] * err
        gradBias[k] += err
      }
    }

    // Gradient ascent update (maximize log-likelihood)
    for (let k = 0; k < nCats; k++) {
      for (let j = 0; j < nFeatures; j++) {
        beta[k][j] += (lr / X.length) * gradBeta[k][j]
      }
      bias[k] += (lr / X.length) * gradBias[k]
    }
  }
  return { beta, bias }
}

// ─── Train once on module load ─────────────────────────────────────────────
let _model = null
function getModel() {
  if (!_model) {
    const { X, y } = generateTrainingData(5000)
    _model = trainLogisticRegression(X, y)
  }
  return _model
}

// ─── Predict ───────────────────────────────────────────────────────────────
// Returns { groceries: 0.68, rent: 0.04, ... }
export function predictRisk(spendArr, budgetArr) {
  const { beta, bias } = getModel()
  const features = [...spendArr.map(v => v / 1000), ...budgetArr.map(v => v / 1000)]
  const probs = {}
  CATEGORIES.forEach((cat, k) => {
    const logit = bias[k] + beta[k].reduce((s, w, j) => s + w * features[j], 0)
    probs[cat] = parseFloat(sigmoid(logit).toFixed(4))
  })
  return probs
}

export function getRiskLevel(prob) {
  if (prob >= 0.65) return { label: 'High',   color: '#ff5566', bg: 'rgba(255,85,102,0.1)',  border: 'rgba(255,85,102,0.25)' }
  if (prob >= 0.35) return { label: 'Medium', color: '#f0a500', bg: 'rgba(240,165,0,0.1)',   border: 'rgba(240,165,0,0.25)' }
  return               { label: 'Low',    color: '#00c4a0', bg: 'rgba(0,196,160,0.1)',   border: 'rgba(0,196,160,0.25)' }
}

export function getHighestRisk(predictions) {
  return Object.entries(predictions).reduce((a, b) => b[1] > a[1] ? b : a)
}

export function getSafest(predictions) {
  return Object.entries(predictions).reduce((a, b) => b[1] < a[1] ? b : a)
}
