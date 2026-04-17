import numpy as np
from scipy.optimize import brentq
from bs_pricer import bs_price
from greeks import vega

def iv_newton(market_price, S, K, T, r, option_type='call',
              sigma_init=0.2, max_iter=100, tol=1e-8):
    
    sigma = sigma_init
    for i in range(max_iter):
        price_model = bs_price(S, K, T, r, sigma, option_type)
        diff = price_model - market_price
        if abs (diff) < tol:
            return sigma
        v = vega (S, K, T, r, sigma) * 100
        if abs(v) < 1e-10:
            return None
        sigma = sigma - diff / v
        if sigma <= 0:
            sigma = 1e-6
    return None

def iv_brent(market_price, S, K, T, r, option_type='call'):
    try:
        objective = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(objective, 1e-4, 10.0, xtol=1e-8)
    except ValueError:
        return None

def implied_vol(market_price, S, K, T, r, option_type='call'):
    iv = iv_newton(market_price, S, K, T, r, option_type)
    if iv is not None and 1e-4 < iv < 10.0:
        return iv
    return iv_brent(market_price, S, K, T, r, option_type)

def vol_smile(S, T, r, strikes, sigma_atm=0.2, skew=-0.08, curvature=0.04):
    log_moneyness = np.log(strikes / S)
    sigmas = sigma_atm + skew * log_moneyness + curvature * log_moneyness**2
    sigmas = np.clip(sigmas, 0.01, 2.0)
    
    market_prices = bs_price(S, strikes, T, r, sigmas, 'call')
    
    implied_vols = np.array([
        implied_vol(p, S, K, T, r, 'call')
        for p, K in zip(market_prices, strikes)
    ])
    
    return {
        'strikes':      strikes,
        'log_moneyness': log_moneyness,
        'sigmas':       sigmas,
        'market_prices': market_prices,
        'implied_vols': implied_vols
    }

if __name__ == "__main__":
    S, K, T, r = 100, 100, 1.0, 0.05

    print("implied vol recovery:")
    for true_sigma in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        market_price = bs_price(S, K, T, r, true_sigma, 'call')
        iv = implied_vol(market_price, S, K, T, r, 'call')
        print(f"  true={true_sigma:.2f}  market price={market_price:.4f}  recovered={iv:.6f}  residual={abs(iv-true_sigma):.2e}")

    print("\nvol smile:")
    strikes = np.linspace(80, 120, 9)
    smile = vol_smile(S=100, T=1.0, r=0.05, strikes=strikes)
    print(f"  {'strike':>8}  {'true sigma':>12}  {'implied vol':>12}")
    for K, s, iv in zip(smile['strikes'], smile['sigmas'], smile['implied_vols']):
        print(f"  {K:>8.0f}  {s*100:>11.1f}%  {iv*100:>11.1f}%")



