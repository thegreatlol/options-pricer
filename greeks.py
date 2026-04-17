import numpy as np
from scipy.stats import norm
from bs_pricer import _d1, _d2, bs_price

def delta(S, K, T, r, sigma, option_type='call'):
    d1 = _d1(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
def gamma(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def theta(S, K, T, r, sigma, option_type='call'):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    if option_type == 'call':
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    elif option_type == 'put':
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    
def rho(S, K, T, r, sigma, option_type='call'):
    d2 = _d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    elif option_type == 'put':
        return -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
def all_greeks(S, K, T, r, sigma, option_type='call'):
    return {
        'delta': delta(S, K, T, r, sigma, option_type),
        'gamma': gamma(S, K, T, r, sigma),
        'vega': vega(S, K, T, r, sigma),
        'theta': theta(S, K, T, r, sigma, option_type),
        'rho': rho(S, K, T, r, sigma, option_type)
    }

def validate_greeks(S, K, T, r, sigma, option_type='call'):
    price = bs_price(S, K, T, r, sigma, option_type)
    greeks = all_greeks(S, K, T, r, sigma, option_type)
    
    # Basic sanity checks
    assert 0 <= greeks['delta'] <= 1 if option_type == 'call' else -1 <= greeks['delta'] <= 0
    assert greeks['gamma'] >= 0
    assert greeks['vega'] >= 0
    assert price >= 0

    deltav = (bs_price(S + .01, K, T, r, sigma, option_type) - bs_price(S - .01, K, T, r, sigma, option_type)) / (2 * .01)
    gamma_v = (bs_price(S + .01, K, T, r, sigma, option_type) - 2 * price + bs_price(S - .01, K, T, r, sigma, option_type)) / (.01 ** 2)
    vega_v = (bs_price(S, K, T, r, sigma + .001, option_type) - bs_price(S, K, T, r, sigma - .001, option_type)) / (2 * .001)
    theta_v = (bs_price(S, K, T + 1/365, r, sigma, option_type) - bs_price(S, K, T - 1/365, r, sigma, option_type)) / (2 * (1/365))
    rho_v = (bs_price(S, K, T, r + .001, sigma, option_type) - bs_price(S, K, T, r - .001, sigma, option_type)) / (2 * .001)

    return {
            'delta residual':  abs(deltav  - greeks['delta']),
            'gamma residual':  abs(gamma_v - greeks['gamma']),
            'vega residual':   abs(vega_v/100 - greeks['vega']),
            'theta residual':  abs(theta_v/365 - greeks['theta']),
            'rho residual':    abs(rho_v/100  - greeks['rho']),
    }

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    print("--- all greeks (ATM call) ---")
    g = all_greeks(S, K, T, r, sigma, 'call')
    for name, val in g.items():
        print(f"  {name:<8}: {val:.6f}")

    print("\n--- validation residuals ---")
    v = validate_greeks(S, K, T, r, sigma, 'call')
    for name, val in v.items():
        print(f"  {name:<20}: {val:.2e}")