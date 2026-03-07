export const CATEGORIES = [
  {
    key: 'groceries',
    label: 'Groceries',
    icon: '🛒',
    color: '#4d9fff',
    // MLE prior defaults (used when user has no history yet)
    defaultMu: 400,
    defaultSigma: 80,
    placeholder: '350',
  },
  {
    key: 'rent',
    label: 'Rent',
    icon: '🏠',
    color: '#a78bfa',
    defaultMu: 1400,
    defaultSigma: 60,
    placeholder: '1400',
  },
  {
    key: 'utilities',
    label: 'Utilities',
    icon: '⚡',
    color: '#f0a500',
    defaultMu: 130,
    defaultSigma: 35,
    placeholder: '120',
  },
  {
    key: 'entertainment',
    label: 'Entertainment',
    icon: '🎬',
    color: '#ff5566',
    defaultMu: 250,
    defaultSigma: 90,
    placeholder: '200',
  },
  {
    key: 'misc',
    label: 'Misc',
    icon: '📦',
    color: '#00c4a0',
    defaultMu: 150,
    defaultSigma: 55,
    placeholder: '130',
  },
]

export const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// Demo data for quick load
export const DEMO_DATA = {
  history: {
    groceries:     [380, 420, 395, 445, 410],
    rent:          [1400, 1400, 1400, 1400, 1400],
    utilities:     [110, 125, 155, 140, 130],
    entertainment: [280, 340, 190, 420, 310],
    misc:          [160, 95, 220, 180, 140],
  },
  budget: {
    groceries: 400,
    rent: 1400,
    utilities: 130,
    entertainment: 250,
    misc: 150,
  },
  midMonth: {
    groceries: 210,
    rent: 700,
    utilities: 65,
    entertainment: 195,
    misc: 90,
  },
  tFraction: 0.5,
}
