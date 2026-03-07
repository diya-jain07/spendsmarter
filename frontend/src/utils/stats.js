/**
 * stats.js — Pure probability math, no external libraries.
 *
 * All CS109 concepts implemented from scratch:
 *   - MLE for Normal distribution parameters
 *   - Bayesian posterior update (Gaussian conjugate)
 *   - Normal CDF via error function approximation
 *   - Sum of independent Normals
 *   - Bernoulli / Binomial PMF
 */

// ─── Normal CDF ────────────────────────────────────────────────────────────
// Approximation of erf via Abramowitz & Stegun formula 7.1.26
// Max error < 1.5e-7 — more than sufficient for this application
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = t * (0.254829592 +
    t * (-0.284496736 +
    t * (1.421413741 +
    t * (-1.453152027 +
    t * 1.061405429))));
  const result = 1 - poly * Math.exp(-x * x);
  return x >= 0 ? result : -result;
}

// Φ(x) — standard normal CDF
export function normalCDF(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

// P(X > x) where X ~ Normal(mu, sigma)
export function probExceed(x, mu, sigma) {
  if (sigma <= 0) return x > mu ? 0 : 1;
  return 1 - normalCDF((x - mu) / sigma);
}

// Normal PDF — for drawing the curve
export function normalPDF(x, mu, sigma) {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// ─── MLE for Normal ────────────────────────────────────────────────────────
// Given samples x_1...x_n:
//   μ̂ = (1/n) Σ x_i
//   σ̂² = (1/n) Σ (x_i - μ̂)²  [MLE estimator, not unbiased]
export function mleFitNormal(samples) {
  if (!samples || samples.length === 0) return { mu: 0, sigma: 1 };
  const n = samples.length;
  const mu = samples.reduce((a, b) => a + b, 0) / n;
  const variance = samples.reduce((a, x) => a + (x - mu) ** 2, 0) / n;
  const sigma = Math.max(Math.sqrt(variance), 1); // floor of 1 to avoid degenerate dist
  return { mu, sigma };
}

// ─── Bayesian Posterior Update ─────────────────────────────────────────────
//
// Prior:       total_spend ~ Normal(mu_prior, sigma_prior²)
// Observation: mid-month spend s observed at fraction t through month
//
// Likelihood model: s = t * total + noise, noise ~ Normal(0, sigma_noise²)
// => s | total ~ Normal(t * total, sigma_noise²)
//
// Posterior precision = prior precision + likelihood precision
//   1/sigma_post² = 1/sigma_prior² + t²/sigma_noise²
//
// Posterior mean:
//   mu_post = sigma_post² * (mu_prior/sigma_prior² + t*s/sigma_noise²)
//
// sigma_noise is estimated as sigma_prior * sqrt(t) — reflecting that
// mid-month spend uncertainty scales with the square root of time elapsed
// (consistent with a random-walk model of daily spending).
//
export function bayesianPosterior(muPrior, sigmaPrior, observedSpend, tFraction) {
  if (tFraction <= 0) return { mu: muPrior, sigma: sigmaPrior };
  if (tFraction >= 1) {
    // Full month observed — posterior collapses to the observation
    return { mu: observedSpend, sigma: Math.max(sigmaPrior * 0.05, 1) };
  }

  // Noise model: spending process has std proportional to sqrt(t)
  const sigmaNoise = Math.max(sigmaPrior * Math.sqrt(tFraction), 1);

  // Precision (1/variance) addition — the CS109 conjugate Gaussian update
  const precPrior = 1 / (sigmaPrior * sigmaPrior);
  const precLikelihood = (tFraction * tFraction) / (sigmaNoise * sigmaNoise);
  const precPost = precPrior + precLikelihood;

  const sigmaPost = Math.sqrt(1 / precPost);
  const muPost = (1 / precPost) * (
    precPrior * muPrior +
    (tFraction / (sigmaNoise * sigmaNoise)) * observedSpend
  );

  return { mu: muPost, sigma: Math.max(sigmaPost, 1) };
}

// ─── Sum of Independent Normals ────────────────────────────────────────────
// If X_k ~ Normal(mu_k, sigma_k²) independently, then
// Σ X_k ~ Normal(Σ mu_k, Σ sigma_k²)
export function sumOfNormals(distributions) {
  const mu    = distributions.reduce((a, d) => a + d.mu, 0);
  const variance = distributions.reduce((a, d) => a + d.sigma * d.sigma, 0);
  return { mu, sigma: Math.sqrt(variance) };
}

// ─── Normal curve points for chart ─────────────────────────────────────────
// Returns array of {x, y} spanning mu ± 3.5*sigma, with n points
export function normalCurvePoints(mu, sigma, n = 120) {
  const lo = mu - 3.5 * sigma;
  const hi = mu + 3.5 * sigma;
  const step = (hi - lo) / (n - 1);
  return Array.from({ length: n }, (_, i) => {
    const x = lo + i * step;
    return { x: Math.round(x * 100) / 100, y: normalPDF(x, mu, sigma) };
  });
}

// ─── Risk level classification ─────────────────────────────────────────────
export function riskLevel(prob) {
  if (prob >= 0.65) return { label: 'High',   color: '#ff5566', bg: 'rgba(255,85,102,0.1)',  border: 'rgba(255,85,102,0.25)' };
  if (prob >= 0.35) return { label: 'Medium', color: '#f0a500', bg: 'rgba(240,165,0,0.1)',   border: 'rgba(240,165,0,0.25)' };
  return               { label: 'Low',    color: '#00c4a0', bg: 'rgba(0,196,160,0.1)',   border: 'rgba(0,196,160,0.25)' };
}

// ─── Binomial PMF ─────────────────────────────────────────────────────────
// P(X = k) where X ~ Binomial(n, p)
function choose(n, k) {
  if (k > n) return 0;
  if (k === 0 || k === n) return 1;
  let result = 1;
  for (let i = 0; i < k; i++) {
    result *= (n - i) / (i + 1);
  }
  return result;
}
export function binomialPMF(n, k, p) {
  return choose(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
}
