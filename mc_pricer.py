import numpy as np
from bs_pricer import bs_price

def simulate_gbm(S, T, r, sigma, n_paths, seed=42, antithetic=False):
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    return S_T


def mc_price(S, K, T, r, sigma, n_paths=100_000, option_type='call', seed=42, antithetic=False):
    S_T = simulate_gbm(S, T, r, sigma, n_paths, seed, antithetic)
    
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S_T, 0)
    
    discounted = np.exp(-r * T) * payoffs
    price      = discounted.mean()
    std_error  = discounted.std() / np.sqrt(n_paths)
    ci_half    = 1.96 * std_error
    
    return price, std_error, ci_half


def mc_convergence(S, K, T, r, sigma, option_type='call',
                   path_counts=None, seed=42):
    if path_counts is None:
        path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    
    bs_ref = bs_price(S, K, T, r, sigma, option_type)
    results = {
        'n_paths':   [],
        'mc_prices': [],
        'ci_lower':  [],
        'ci_upper':  [],
        'bs_price':  bs_ref
    }
    
    for n in path_counts:
        price, _, ci = mc_price(S, K, T, r, sigma, n, option_type, seed)
        results['n_paths'].append(n)
        results['mc_prices'].append(price)
        results['ci_lower'].append(price - ci)
        results['ci_upper'].append(price + ci)
    
    return results

if __name__ == "__main__":
    S_T = simulate_gbm(100, 1.0, 0.05, 0.2, n_paths=100_000)
    print(f"mean of S_T : {S_T.mean():.4f}")
    print(f"std of S_T  : {S_T.std():.4f}")

    bs_ref = bs_price(100, 100, 1.0, 0.05, 0.2, 'call')
    print(f"\nBS price    : {bs_ref:.6f}")

    price, se, ci = mc_price(100, 100, 1.0, 0.05, 0.2, antithetic=False)
    print(f"\nwithout antithetic:")
    print(f"  price     : {price:.6f}  error: {abs(price - bs_ref):.6f}")
    print(f"  std error : {se:.6f}  95% CI: ±{ci:.6f}")

    price, se, ci = mc_price(100, 100, 1.0, 0.05, 0.2, antithetic=True)
    print(f"\nwith antithetic:")
    print(f"  price     : {price:.6f}  error: {abs(price - bs_ref):.6f}")
    print(f"  std error : {se:.6f}  95% CI: ±{ci:.6f}")

    conv = mc_convergence(100, 100, 1.0, 0.05, 0.2)
    print(f"\nconvergence study:")
    for n, p in zip(conv['n_paths'], conv['mc_prices']):
        print(f"  N={n:<8}  price={p:.4f}  error={abs(p - conv['bs_price']):.4f}")