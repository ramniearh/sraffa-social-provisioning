import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    [ 50, 200, 100,   0,   0,1630,   0,  10,   0,  10,   0], # 1. Grain 
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

# -- Figure 1: CUSTOM Multi-Color Physical Matrix Heatmap --
fig1, ax1 = plt.subplots(figsize=(14, 10))
plt.title("Physical Matrix Omega: Custom Row-Normalized Flows", fontsize=16, pad=20)

# Build a custom RGBA image so each row has its own color gradient
rgba_matrix = np.ones((N, N+1, 4)) 
base_colors = plt.cm.tab20.colors 

for i in range(N):
    row_color = np.array(base_colors[i % 20])
    row_max = np.max(omega_with_total[i])
    for j in range(N+1):
        val = omega_with_total[i, j]
        # Calculate opacity (alpha) based on row's max value to show relative distribution
        alpha = 0.0
        if val > 0:
            alpha = 0.15 + 0.85 * (val / row_max) # Base opacity so low numbers aren't invisible
            
        rgba_matrix[i, j, :3] = row_color
        rgba_matrix[i, j, 3] = alpha
        
        # Add text annotations
        text_color = "black" if alpha < 0.65 else "white"
        ax1.text(j, i, f"{int(val)}", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

ax1.imshow(rgba_matrix, aspect='auto')

# Formatting the custom plot
ax1.set_xticks(np.arange(N+1))
ax1.set_yticks(np.arange(N))
ax1.set_xticklabels(heatmap_cols, rotation=45, ha='right')
ax1.set_yticklabels(sectors)
# Remove grid lines for a cleaner look
ax1.grid(False) 
plt.tight_layout()

# -- Figure 2: Technical Matrix Heatmap (Kept Uniform to reflect Normalization) --
fig2 = plt.figure(figsize=(14, 10))
plt.title("Technical Matrix A: The Recipes (Scale Removed)", fontsize=16, pad=20)
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

# -- Figure 4: The Eigenvector (Sraffian Prices as a Radar Chart) --
fig4 = plt.figure(figsize=(12, 12))
ax = fig4.add_subplot(111, polar=True)
plt.title("The Dominant Eigenvector: Absolute Prices\n(Geometric Shape = Exchange Ratios)", fontsize=16, pad=40)

# 1. Math Prep: Radar charts need the data to loop back to the start to close the polygon
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the circle for the angles
prices_loop = prices.tolist() + [prices[0]]  # Close the circle for the data

# 2. Draw the underlying web (The Eigenvector shape)
ax.plot(angles, prices_loop, color='gray', linewidth=2, linestyle='solid', zorder=1)
ax.fill(angles, prices_loop, color='lightgray', alpha=0.3, zorder=0)

# 3. Plot the nodes using the EXACT colors from the Physical Matrix (Figure 1)
base_colors = plt.cm.tab20.colors 
for i in range(N):
    marker_color = base_colors[i % 20]
    # Make the Numeraire pop out visually
    is_num = (i == numeraire_index)
    size = 15 if is_num else 10
    edge = 'red' if is_num else 'black'
    edge_width = 2.5 if is_num else 1
    
    ax.plot(angles[i], prices[i], marker='o', markersize=size, color=marker_color, 
            markeredgecolor=edge, markeredgewidth=edge_width, zorder=2)
    
    # Add numerical price labels slightly offset from the nodes
    ax.text(angles[i], prices[i] + (max(prices)*0.08), f"{prices[i]:.2f}", 
            ha='center', va='center', fontsize=9, fontweight='bold' if is_num else 'normal')

# 4. Format the Axes & Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(sectors, fontsize=11)

# Color the text labels to match the nodes, and bold the Numeraire
for i, label in enumerate(ax.get_xticklabels()):
    label.set_color(base_colors[i % 20])
    if i == numeraire_index:
        label.set_fontweight('bold')
        label.set_color('red')

# 5. Add Explanatory Legend/Text
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(1.1, 0.05, f"Numeraire (Anchor):\n1 Unit of {sectors[numeraire_index]}\n= 1.0 PDU\n(Highlighted in Red)", 
        transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)

plt.tight_layout()

# Show all figures
plt.show()


# =====================================================================
# THEORETICAL FOUNDATIONS & FUTURE RESEARCH EXTENSIONS
# =====================================================================
"""
NOTE 1: THE GOVERNANCE OVERLAY MATRIX (Gamma) & WILLIAMSON
- Future ABM steps: Create a second matrix `gamma` of same dimensions.
- Populate it with institutional states: 'M' (Market), 'F' (Firm), 'S' (State), 'C' (Communal).
- Algorithm idea: If an `omega` block shows high internal density but low external edges 
  (high asset specificity), transaction cost logic should automatically force its `gamma` 
  state from 'M' to 'F' (corporate consolidation).

NOTE 2: N-K DYNAMICS (Stuart Kauffman)
- N = Matrix dimension. K = Density of the `omega` matrix.
- Innovation in an ABM is an evolutionary search: slightly mutate a coefficient a_ij.
- If this mutation lowers total thermodynamic respiration (or increases the Eigenvalue > 1), 
  the technology spreads.

NOTE 3: BIOLOGICAL SANDBOXING
- Because the math is agnostic, renaming sectors allows ecological modeling.
- [Workers -> Fungi, Grain -> Oak Canopy, Tools -> Nitrogen, etc.]
- The matrix structural blocks change from 'Firms' to 'Trophic Cascades' and 'Symbiotic Mutualisms'.

NOTE 4: SIMMEL, ABSTRACTION, & THE PDU
- The prices (PDUs) calculated here represent Simmel's "claims upon society" or 
  "the value of things without the things themselves".
- By imposing the Numeraire, the Authority converts physical interdependence into quantified social trust.

NOTE 5: THE POLANYIAN SURPLUS SHOCK (The Bridge to Capitalism)
- Current state: Eigenvalue = 1 (Perfect Subsistence).
- Next step: Introduce a technology shock. Grain output jumps to 2500, inputs remain identical.
- The Eigenvalue drops below 1 (the system produces more than it consumes). 
- How is the surplus distributed?
    a) RECIPROCITY: Absorb it into 'Rituals' (Column 9).
    b) REDISTRIBUTION: Absorb it into 'Authority' (Column 7) to build Non-Basics.
    c) EXCHANGE: Rip the 'Workers' row/col out, establish a Wage (w) and Profit Rate (r).

NOTE 6: EXTERNAL TRADE & OPEN SYSTEMS
- This is currently a closed thermodynamic loop. To model early trade (e.g., Bronze Age imports), 
  we must append an 'Export' column and 'Import' row.

NOTE 7: THEORETICAL FOUNDATIONS & SUGGESTED READING
- Anthropology/Gift Economies: Chris Gregory (1982) "Gifts and Commodities". Proves that in clan-based
  societies, surplus is not accumulated as profit, but is systematically converted into social 
  debts/relations via structured gifting rituals.
- Feminist Economics/Social Reproduction: Antonella Picchio (1992) "Social Reproduction". Formalizes 
  the concept that labor does not organically appear on the market; it is actively produced by an immense, 
  unpriced "Household/Care" sector that structurally subsidizes the formal capitalist matrix.
- Institutionalism/Pre-Capitalist Surplus: Sergio Cesaratto (2015/2020). Bridges Sraffa with Karl Polanyi, 
  demonstrating that in societies without a uniform profit rate, the physical surplus is distributed via 
  institutional forces (Redistribution/Reciprocity) such as the State or Temple.
- Ecological I-O: Bruce Hannon (1973) "The Structure of Ecosystems". Strips away human assumptions entirely, 
  proving that ecosystems (tracking solar energy/carbon flows) adhere strictly to Sraffian topological laws.

NOTE 8: ENTROPY & DISSIPATIVE STRUCTURES (Prigogine)
- In Sraffa's system, surplus is either perfectly consumed or extracted, allowing the cycle to restart.
- Through a thermodynamic lens, the Sraffian matrix is a "Dissipative Structure" operating far from equilibrium.
- A surplus represents an injection of "free energy" (exergy). If the society does not continuously 
  dissipate that energy through expanded consumption, luxury, or warfare (acting as entropy sinks), 
  the system faces a thermodynamic/accumulation crisis and structurally collapses.
"""