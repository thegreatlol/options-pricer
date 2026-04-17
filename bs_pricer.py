import numpy as np 
from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / ( sigma * np.sqrt(T) )
    return d1

def _d2(S, K, T, r, sigma):
    d2 = _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    return d2

def bs_call_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def bs_put_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    put_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def bs_price(S, K, T, r, sigma, option_type='call'):
    if option_type == 'call':
        return bs_call_price(S, K, T, r, sigma)
    elif option_type == 'put':
        return bs_put_price(S, K, T, r, sigma)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def put_call_parity_check(S, K, T, r, sigma):
    C = bs_call_price(S, K, T, r, sigma)
    P = bs_put_price(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return {
        'C - P':    lhs,
        'S - PV(K)': rhs,
        'residual': abs(lhs - rhs)
    }

def intrinsic_value(S, K, option_type='call'):
    if option_type == 'call':
        return np.maximum(S - K, 0)
    else:
        return np.maximum(K - S, 0)

if __name__ == "__main__":
    print(_d1(100, 100, 1.0, 0.05, 0.2))
    print(_d2(100, 100, 1.0, 0.05, 0.2))
    print(bs_call_price(100, 100, 1.0, 0.05, 0.2))
    print(bs_put_price(100, 100, 1.0, 0.05, 0.2))
    print(bs_price(100, 100, 1.0, 0.05, 0.2, option_type='call'))
    print(put_call_parity_check(100, 100, 1.0, 0.05, 0.2))
    print(intrinsic_value(110, 100, 'call'))  # expect 10
    print(intrinsic_value(90, 100, 'call'))   # expect 0
    print(intrinsic_value(90, 100, 'put'))    # expect 10
    print(intrinsic_value(110, 100, 'put'))   # expect 0