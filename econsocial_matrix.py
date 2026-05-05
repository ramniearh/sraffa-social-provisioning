import numpy as np
import pandas as pd
import scipy.linalg as la

# 1. Define the Sectors by Institutional Boundary (N = 9)
# Zone M: Market (Commodities, low asset specificity)
# Zone O: Organization (State, planning, high asset specificity)
# Zone F: Family/Community (Care, infinite asset specificity/non-fungible)

sectors = [
    'Food (M)', 'Energy (M)', 'Tools (M)',             # Market Zone
    'Infrastructure (O)', 'State Admin/Taxes (O)',     # Organization Zone
    'Care Work (F)', 'Education (F)', 'Human Cap (F)'  # Family/Community Zone
]

# 2. Build the Block Matrix (Columns = Recipes, Rows = Allocations)
# Note the off-diagonal flows (e.g., Care flowing into Market, Taxes into Family)
A_raw = np.array([
    #F(M)  E(M)  T(M) | I(O)  S(O) | C(F)  Ed(F) H(F) 
    [0.10, 0.05, 0.05,  0.05, 0.05,  0.15, 0.10, 0.20], # Food (M)
    [0.05, 0.15, 0.15,  0.10, 0.10,  0.05, 0.05, 0.10], # Energy (M)
    [0.10, 0.20, 0.15,  0.20, 0.05,  0.00, 0.05, 0.00], # Tools (M)
    #--------------------------------------------------
    [0.05, 0.10, 0.10,  0.10, 0.10,  0.05, 0.10, 0.05], # Infrastructure (O)
    [0.10, 0.10, 0.10,  0.00, 0.00,  0.00, 0.00, 0.10], # State/Taxes (O)
    #--------------------------------------------------
    [0.00, 0.00, 0.00,  0.00, 0.00,  0.20, 0.10, 0.30], # Care Work (F)
    [0.05, 0.05, 0.10,  0.10, 0.20,  0.10, 0.10, 0.20], # Education (F)
    [0.30, 0.20, 0.30,  0.40, 0.45,  0.35, 0.40, 0.00]  # Human Capability (F)
])

# 3. Normalize to a Subsistence/Self-Replacing State (Dom Eigenvalue = 1)
# This represents a society that can exactly reproduce its material and social base.
eigenvals = np.linalg.eigvals(A_raw)
dom_eigenval = np.max(np.real(eigenvals))
A = A_raw / dom_eigenval

# 4. Ingham/Pivetti Monetary Closure
# The Central Bank (Zone O) sets the interest rate (i), determining the profit rate (r).
i = 0.04  # 4% Interest Rate
rho = 0.02 # 2% Risk Premium for Market Sectors
r = i + rho

# 5. Solve for the Shadow/Sraffian Prices
# Equation: p = (1+r)pA  =>  p[I - (1+r)A] = 0
I = np.eye(len(sectors))
M_matrix = I - (1+r) * A

# Use Singular Value Decomposition to find the null space (left eigenvector)
U, S, Vh = la.svd(M_matrix)
prices = np.abs(Vh[-1, :]) 

# Nominally anchor the system: 1 unit of Tax Clearance = $1.00 Unit of Account
tax_index = sectors.index('State Admin/Taxes (O)')
prices = prices / prices[tax_index]

# 6. Output Analysis
df = pd.DataFrame({
    'Institutional Zone': [s.split(' ')[-1] for s in sectors],
    'Sector': [s.split(' (')[0] for s in sectors],
    'Sraffian Shadow Price (Tax = $1)': np.round(prices, 2)
})

print("--- UNIFIED SRAFFIAN/POLANYIAN BLOCK MODEL ---")
print(f"Central Bank Interest Rate (i): {i*100}%")
print(f"Resulting Matrix Profit Rate (r): {r*100}%\n")
print(df.to_string(index=False))
print("\nNote: 'Shadow Prices' for Zone F represent the objective material/social cost")
print("required to provision human capability, regardless of market wages.")
