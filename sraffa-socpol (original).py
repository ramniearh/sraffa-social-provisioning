import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# =====================================================================
# THE SOCIO-ECONOMIC SUBSISTENCE MATRIX (Zero Surplus)
# "Reproduction of society by means of society"
# =====================================================================

sectors = [
    "0. Workers (labor-hours)",  
    "1. Grain (tons)",           
    "2. Energy (MWh)",           
    "3. Metal (tons)",           
    "4. Tools (units)",          
    "5. Food Rations (units)",   
    "6. Housing (units)",        
    "7. Authority (units)",      
    "8. Infrastructure (units)", 
    "9. Communal Rituals",       
    "10. Consumer Goods"         
]

N = len(sectors)

# Physical input-output matrix (Rows = Distribution, Columns = Recipes)
omega = np.array([
    #W    G    E    M    T    F    H    A    I    R    CG
    [300, 500, 300, 400, 200, 400, 300, 100, 300,  50, 150], # 0. Workers
    [ 0, 200, 100,   0,   0,1630,   0,  10,   0,  10,   0], # 1. Grain (ADJUSTED: Rituals down, Food up)
    [ 50, 100, 100, 200, 100, 200, 100,  20,  50,  30,  50], # 2. Energy
    [  0,  50, 100,  50, 150,   0,  50,   0, 100,   0,   0], # 3. Metal
    [ 50,  50,  20,  30,  20,  30,  40,  10,  30,  10,  10], # 4. Tools
    [800, 100,  50, 100, 100, 100, 100,  50,  50,   0,  50], # 5. Food Rations
    [150,  50,  50,  50,  50,  50,  20,  10,  20,  20,  30], # 6. Housing
    [ 10,  10,  10,  10,  10,  10,  10,   5,  15,   5,   5], # 7. Authority
    [ 20,  20,  30,  40,  20,  20,  20,  10,  10,   5,   5], # 8. Infrastructure
    [ 50,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], # 9. Rituals
    [400,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]  # 10. Consumer Goods 
], dtype=float)

total_output = np.array([3000, 2000, 1000, 500, 300, 1500, 500, 100, 200, 50, 400])

omega_with_total = np.column_stack((omega, total_output))
heatmap_cols = sectors + ["TOTAL OUTPUT"]

# =====================================================================
# MODULE 1: MATHEMATICAL CALCULATIONS
# =====================================================================

A = np.zeros((N, N))
for j in range(N):
    A[:, j] = omega[:, j] / total_output[j]

eigenvalues, eigenvectors = np.linalg.eig(A.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
p_raw = np.abs(np.real(eigenvectors[:, idx]))

# Numeraire
numeraire_index = 7 # Authority
prices = p_raw / p_raw[numeraire_index]

# =====================================================================
# MODULE 2: TERMINAL REPORT
# =====================================================================

print("\n" + "="*70)
print(" SRAFFIAN SOCIO-ECONOMIC REPORT ")
print("="*70)

print("\n1. PHYSICAL MATRIX (Omega) & TOTAL OUTPUT")
print("-" * 70)
header = f"{'Sector':<25}" + "".join([f"{i:>5}" for i in range(N)]) + " | Total"
print(header)
for i in range(N):
    row_str = f"{sectors[i][:23]:<25}" + "".join([f"{int(x):>5}" for x in omega[i]]) + f" | {int(total_output[i]):>5}"
    print(row_str)

print("\n2. TECHNICAL MATRIX (A) - Unit Recipes")
print("-" * 70)
header_A = f"{'Sector':<25}" + "".join([f"{i:>6}" for i in range(N)])
print(header_A)
for i in range(N):
    row_str = f"{sectors[i][:23]:<25}" + "".join([f"{x:>6.3f}" for x in A[i]])
    print(row_str)

print("\n3. LINEAR ALGEBRA (Eigendecomposition)")
print("-" * 70)
print("All Calculated Eigenvalues:")
for i, ev in enumerate(eigenvalues):
    marker = " <-- (Subsistence Break-Even)" if i == idx else ""
    print(f"  λ_{i}: {np.real(ev):.5f} {marker}")

print(f"\nRaw Eigenvector (Relative Prices) for λ ≈ 1.0:")
print(np.array2string(p_raw, formatter={'float_kind':lambda x: f"{x:.4f}"}))

print("\n4. ABSOLUTE PRICES (Applying the Numeraire)")
print("-" * 70)
print(f"UNIT OF ACCOUNT: 1 Unit of {sectors[numeraire_index]} = 1.00 PDU\n")
for i in range(N):
    print(f"{sectors[i]:<27}: {prices[i]:>8.4f} PDUs")
print("="*70 + "\n")

# =====================================================================
# MODULE 3: VISUALIZATIONS (Separate Popups)
# =====================================================================
sns.set_theme(style="whitegrid")

# -- Figure 1: Physical Matrix Heatmap --
fig1 = plt.figure(figsize=(14, 10))
plt.title(r"Physical Matrix $\Omega$: Aggregate Flows & Scale", fontsize=16, pad=20)
sns.heatmap(omega_with_total, annot=True, fmt=".0f", cmap="YlOrRd", 
            xticklabels=heatmap_cols, yticklabels=sectors, 
            cbar_kws={'label': 'Physical Units'})
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# -- Figure 2: Technical Matrix Heatmap --
fig2 = plt.figure(figsize=(14, 10))
plt.title(r"Technical Matrix $A$: The Recipes (Scale Removed)", fontsize=16, pad=20)
sns.heatmap(A, annot=True, fmt=".3f", cmap="Blues", 
            xticklabels=sectors, yticklabels=sectors,
            cbar_kws={'label': 'Coefficient (Input per 1 Output)'})
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# -- Figure 3: Network Topology --
fig3 = plt.figure(figsize=(12, 10))
plt.title("Network Topology: Interdependence Hubs", fontsize=16)
G = nx.from_numpy_array(omega, create_using=nx.DiGraph)
color_map = ['lightblue'] + ['lightgreen']*6 + ['salmon']*3 + ['lightgreen']
pos = nx.spring_layout(G, k=0.9, seed=42)
edges = G.edges()
weights = [np.log1p(G[u][v]['weight']) * 0.5 for u, v in edges]
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=800, edgecolors='black')
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrowsize=12, alpha=0.4, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_labels(G, pos, {i: sectors[i].split('. ')[1] for i in range(N)}, font_size=9, font_weight="bold")
plt.axis('off')
plt.tight_layout()

# -- Figure 4: The Eigenvector (Sraffian Prices) --
fig4 = plt.figure(figsize=(12, 8))
plt.title("The Dominant Eigenvector: Absolute Prices (Anchored by Authority)", fontsize=16, pad=20)

# Create color array to highlight the Numeraire
bar_colors = ['#4C72B0' if i != numeraire_index else '#C44E52' for i in range(N)]

ax = sns.barplot(x=prices, y=sectors, palette=bar_colors)
ax.set_xlabel("Price in Palace-Debt-Units (PDUs)", fontsize=12)
ax.set_ylabel("Socio-Economic Sector", fontsize=12)

# Add value labels to the end of the bars
for i, p in enumerate(prices):
    ax.text(p + (max(prices)*0.01), i, f"{p:.2f} PDU", va='center', fontsize=10, fontweight='bold' if i == numeraire_index else 'normal')

# Add a text box explaining the Numeraire
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.65, 0.1, f"Numeraire (Anchor):\n1 Unit of {sectors[numeraire_index]}\n= 1.0 PDU\n\n(Highlighted in Red)", 
        transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)

plt.tight_layout()

# Show all figures
plt.show()

# =====================================================================
# THEORETICAL FOUNDATIONS & FUTURE RESEARCH EXTENSIONS
# =====================================================================
"""
[... Theoretical Notes 1-7 Preserved ...]
"""