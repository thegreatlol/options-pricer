import numpy as np
import matplotlib.pyplot as plt
from bs_pricer import bs_price, intrinsic_value
from greeks import delta, gamma, vega, theta
from mc_pricer import mc_convergence
from iv_solver import vol_smile

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "font.size":         10,
})

def plot_price_decomposition(K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'):
    S_range   = np.linspace(60, 140, 300)
    price     = bs_price(S_range, K, T, r, sigma, option_type)
    intrinsic = intrinsic_value(S_range, K, option_type)
    time_val  = price - intrinsic

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(S_range, price, lw=2, label='Option price')
    ax.fill_between(S_range, 0, intrinsic, alpha=0.25, label='Intrinsic value')
    ax.fill_between(S_range, intrinsic, price, alpha=0.25, label='Time value')
    ax.axvline(K, color='gray', lw=1, ls=':', alpha=0.7)
    ax.set_xlabel('Spot price S')
    ax.set_ylabel('Value ($)')
    ax.set_title(f'European {option_type} — price decomposition')
    ax.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig('fig1_price_decomposition.png', dpi=150)
    plt.show()


def plot_greeks_surface(K=100, r=0.05, sigma=0.2, option_type='call'):
    S_range = np.linspace(70, 130, 80)
    T_range = np.linspace(0.05, 2.0, 80)
    S_grid, T_grid = np.meshgrid(S_range, T_range)

    greeks = {
        'Delta': delta(S_grid, K, T_grid, r, sigma, option_type),
        'Gamma': gamma(S_grid, K, T_grid, r, sigma),
        'Vega':  vega(S_grid, K, T_grid, r, sigma),
        'Theta': theta(S_grid, K, T_grid, r, sigma, option_type),
    }

    fig = plt.figure(figsize=(14, 10))
    for idx, (name, values) in enumerate(greeks.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.plot_surface(S_grid, T_grid, values, alpha=0.85, linewidth=0, antialiased=True)
        ax.set_xlabel('Spot S', fontsize=8)
        ax.set_ylabel('Time T', fontsize=8)
        ax.set_zlabel(name, fontsize=8)
        ax.set_title(name, fontsize=11)
        ax.view_init(elev=22, azim=-50)
        ax.tick_params(labelsize=7)

    plt.suptitle(f'Greeks surfaces — European {option_type}', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig2_greeks_surfaces.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_mc_convergence(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'):
    results = mc_convergence(S, K, T, r, sigma, option_type)
    
    n      = np.array(results['n_paths'])
    prices = np.array(results['mc_prices'])
    lo     = np.array(results['ci_lower'])
    hi     = np.array(results['ci_upper'])
    bs_ref = results['bs_price']
    errors = np.abs(prices - bs_ref)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.fill_between(n, lo, hi, alpha=0.25, label='95% CI')
    ax1.plot(n, prices, lw=2, marker='o', ms=4, label='MC price')
    ax1.axhline(bs_ref, lw=1.5, ls='--', label=f'BS price = {bs_ref:.4f}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of paths (log scale)')
    ax1.set_ylabel('Option price ($)')
    ax1.set_title('MC convergence to BS price')
    ax1.legend(framealpha=0)

    ref_line = errors[0] * np.sqrt(n[0]) / np.sqrt(n)
    ax2.plot(n, errors,   lw=2, marker='o', ms=4, label='|MC - BS|')
    ax2.plot(n, ref_line, lw=1.5, ls=':',         label='1/√N reference')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of paths (log scale)')
    ax2.set_ylabel('Absolute error (log scale)')
    ax2.set_title('Convergence rate: O(1/√N)')
    ax2.legend(framealpha=0)

    plt.suptitle('Monte Carlo convergence', fontsize=11)
    plt.tight_layout()
    plt.savefig('fig3_mc_convergence.png', dpi=150)
    plt.show()

def plot_vol_surface(S=100, r=0.05):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    subplot_kw={'projection': None})
    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    strikes = np.linspace(75, 125, 30)
    maturities = np.linspace(0.1, 2.0, 20)

    for T, label in [(0.25, 'T=3m'), (0.5, 'T=6m'), (1.0, 'T=1y'), (2.0, 'T=2y')]:
        skew = -0.15 / np.sqrt(T)   # skew steepens for shorter maturities
        smile = vol_smile(S=S, T=T, r=r, strikes=strikes, skew=skew)
        ivs = np.array([iv if iv is not None else np.nan
                        for iv in smile['implied_vols']])
        ax1.plot(strikes, ivs * 100, lw=1.8, label=label)

    ax1.axvline(S, color='gray', lw=1, ls=':', alpha=0.7)
    ax1.set_xlabel('Strike K')
    ax1.set_ylabel('Implied volatility (%)')
    ax1.set_title('Volatility smile')
    ax1.legend(framealpha=0)

    K_grid, T_grid = np.meshgrid(strikes, maturities)
    IV_grid = np.zeros_like(K_grid)
    for i, T in enumerate(maturities):
        skew = -0.15 / np.sqrt(T)
        smile = vol_smile(S=S, T=T, r=r, strikes=strikes, skew=skew)
        for j, iv in enumerate(smile['implied_vols']):
            IV_grid[i, j] = iv * 100 if iv is not None else np.nan

    ax2.plot_surface(K_grid, T_grid, IV_grid, cmap='coolwarm',
                     alpha=0.9, linewidth=0, antialiased=True)
    ax2.set_xlabel('Strike K', fontsize=8)
    ax2.set_ylabel('Maturity T', fontsize=8)
    ax2.set_zlabel('IV (%)', fontsize=8)
    ax2.set_title('Implied vol surface', fontsize=11)
    ax2.view_init(elev=25, azim=-60)
    ax2.tick_params(labelsize=7)

    plt.suptitle('Implied volatility — smile and surface', fontsize=11)
    plt.tight_layout()
    plt.savefig('fig4_vol_surface.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
        plot_price_decomposition()
        plot_greeks_surface()
        plot_mc_convergence()
        plot_vol_surface()