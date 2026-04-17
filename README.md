# Options Pricing Engine

A Python implementation of European option pricing via Black-Scholes analytics, 
Monte Carlo simulation, and implied volatility inversion.

---

## Modules

| File | Description |
|---|---|
| `bs_pricer.py` | Analytical Black-Scholes pricer for European calls and puts, with put-call parity verification |
| `greeks.py` | All five Greeks analytically, validated against finite differences to 1e-9 |
| `mc_pricer.py` | Monte Carlo pricer with antithetic variates and convergence study |
| `iv_solver.py` | Implied vol inversion via Newton-Raphson with Brent fallback, synthetic vol smile |
| `visualizations.py` | Four publication-quality figures |


