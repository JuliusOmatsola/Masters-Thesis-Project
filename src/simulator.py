
import numpy as np
def revenue_for_price_grid(elasticity, C, price_grid):
    # demand = C * p^elasticity
    demand = C * (price_grid ** elasticity)
    revenue = price_grid * demand
    return demand, revenue

def find_revenue_max_price(elasticity_est, elasticity_boots, C, p0,
                           floor_price=None, ceil_price=None,
                           down_pct=0.3, up_pct=0.2, step=0.01):
    # define grid
    if floor_price is None:
        floor_price = p0*(1-down_pct)
    if ceil_price is None:
        ceil_price = p0*(1+up_pct)
    grid = np.arange(floor_price, ceil_price+1e-9, step)
    # point estimate
    _, rev = revenue_for_price_grid(elasticity_est, C, grid)
    best_idx = np.nanargmax(rev)
    best_price = grid[best_idx]
    best_revenue = rev[best_idx]
    # uncertainty: compute revenue for many bootstrap elasticities and get quantiles of revenue-max price
    best_prices_boot = []
    for b in elasticity_boots:
        _, revb = revenue_for_price_grid(b, C, grid)
        best_prices_boot.append(grid[np.nanargmax(revb)])
    lower = np.percentile(best_prices_boot, 2.5)
    upper = np.percentile(best_prices_boot, 97.5)
    return {'best_price':best_price, 'best_revenue':best_revenue, 'price_ci':(lower,upper), 'grid':grid, 'rev_grid':rev}
