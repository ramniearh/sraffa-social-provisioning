import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx

# =====================================================================
# SRAFFIAN MATRIX OF PROVISIONING & SOCIAL REPRODUCTION
# A unified formalism for social and economic provisioning systems,
# expanding Sraffa's 'production of commodities by means of 
# commodities' framework to account for social reproduction and
# macro-institutions (chartalism, feminism)
# =====================================================================

# 8-Sector Aggregation: Concrete physical sectors + Social/Institutional sinks
sectors = [
    "0. Grain (tons)",           
    "1. Energy (MWh)",
    "2. Raw Materials (tons)",           
    "3. Tools & Implements (units)",          
    "4. Food Rations (units)",   
    "5. Shelter & Housing (units)",        
    "6. Workers & Reproduction (hours)",         
    "7. Authority, Taxes & Infrastructure"   
]

N = len(sectors)

# Custom Color Palette mapped to the physical nature of the sectors
hex_colors = [
    '#DEB887', # 0. Grain (Wheat/Tan)
    '#FFD700', # 1. Energy (Gold/Yellow)
    '#B0C4DE', # 2. Raw Materials (Light Steel Blue/Grey)
    '#708090', # 3. Tools (Slate Grey)
    '#F4A460', # 4. Food Rations (Warm Brown/Orange)
    '#CD5C5C', # 5. Shelter & Housing (Terracotta/Brick)
    '#2E8B57', # 6. Workers (Forest Green)
    '#6A5ACD'  # 7. Authority (Deep Purple)
]
custom_colors = [mcolors.to_rgb(c) for c in hex_colors]

# The Oikos Matrix (Rows = Physical Distribution, Columns = Input Recipes)
oikos = np.array([
    #G    E    RM   T    Food Hse  Work Auth 
    [200, 100,   0,   0,1630,   0,  60,  10], # 0. Grain
    [100, 100, 200, 100, 250, 100,  80,  70], # 1. Energy
    [ 50, 100,  50, 150,   0,  50,   0, 100], # 2. Raw Materials
    [ 50,  20,  30,  20,  40,  40,  60,  40], # 3. Tools
    [100,  50, 100, 100, 150, 100, 800, 100], # 4. Food Rations
    [ 50,  50,  50,  50,  80,  20, 170,  30], # 5. Shelter & Housing
    [500, 300, 400, 200, 550, 300, 400, 400], # 6. Workers & Reproduction
    [ 30,  40,  50,  30,  50,  30,  40,  30]  # 7. Authority, Taxes & Infra.
], dtype=float)

# Gross output produced by each sector per cycle
total_output = np.array([2000, 1000, 500, 300, 1500, 500, 3050, 300], dtype=float)

# For visualization purposes
oikos_with_total = np.column_stack((oikos, total_output))
heatmap_cols = sectors + ["TOTAL OUTPUT"]

# =====================================================================
# MODULE 1: SYSTEM DIAGNOSTICS & SUBSISTENCE CHECK
# =====================================================================

print("\n" + "="*70)
print(" PROVISIONING MATRIX: DIAGNOSTICS REPORT ")
print("="*70)

# 1. Physical Balance (Accounting Surplus/Deficit)
row_sums = np.sum(oikos, axis=1)
net_surplus = total_output - row_sums

print("\n--- 1. PHYSICAL SURPLUS / DEFICIT ---")
is_balanced = True
for i in range(N):
    if net_surplus[i] > 1e-5:
        print(f"  [{sectors[i]}] SURPLUS: +{net_surplus[i]:.2f} units")
        is_balanced = False
    elif net_surplus[i] < -1e-5:
        print(f"  [{sectors[i]}] DEFICIT: {net_surplus[i]:.2f} units (WARNING)")
        is_balanced = False
        
if is_balanced:
    print("  All sectors are in perfect physical balance (Net 0).")

# 2. Technical Coefficients Matrix (A)
A = np.zeros((N, N))
for j in range(N):
    if total_output[j] > 0:
        A[:, j] = oikos[:, j] / total_output[j]

# 3. Topological Evaluation (Perron-Frobenius Root)
eigenvalues, eigenvectors = np.linalg.eig(A.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
dom_lambda = np.real(eigenvalues[idx])

print("\n--- 2. TOPOLOGICAL EIGENVALUE (λ) ---")
print(f"  Dominant λ: {dom_lambda:.5f}")

# Halt execution if the system is not at strict subsistence
if not np.isclose(dom_lambda, 1.0, atol=1e-4):
    print("\n>>> SYSTEM HALT: Matrix is not at Subsistence (λ ≠ 1.0). <<<")
    print(">>> Adjust the inputs in the oikos matrix or total_output array to balance the system.")
    print(">>> Visualizations and absolute prices require a balanced matrix.")
    sys.exit() 

# 4. Relative Prices (Dominant Eigenvector)
p_raw = np.abs(np.real(eigenvectors[:, idx]))

# Apply Numeraire (Anchor unit of account)
numeraire_index = 7 # Anchor set to Authority
prices = p_raw / p_raw[numeraire_index]


# =====================================================================
# MODULE 2: DATA EXPORT
# =====================================================================

print("\n--- 3. THE OIKOS MATRIX (Raw Physical Units) ---")
header = f"{'Sector':<40}" + "".join([f"{i:>6}" for i in range(N)]) + " | Total"
print(header)
for i in range(N):
    row_str = f"{sectors[i][:38]:<40}" + "".join([f"{int(x):>6}" for x in oikos[i]]) + f" | {int(total_output[i]):>5}"
    print(row_str)

print("\n--- 4. ABSOLUTE PRICES (Applying Numeraire) ---")
print(f"UNIT OF ACCOUNT: 1 Unit of {sectors[numeraire_index]} = 1.00 PDU\n")
for i in range(N):
    print(f"{sectors[i]:<40}: {prices[i]:>8.4f} PDUs")
print("="*70 + "\n")


# =====================================================================
# MODULE 3: VISUALIZATIONS 
# =====================================================================
sns.set_theme(style="whitegrid")

# -- Figure 1: Physical Matrix Flows --
fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.title("Physical Matrix: Aggregate Intra-System Flows", fontsize=16, pad=40)

rgba_matrix = np.ones((N, N+1, 4)) 
for i in range(N):
    row_color = np.array(custom_colors[i])
    row_max = np.max(oikos_with_total[i])
    for j in range(N+1):
        val = oikos_with_total[i, j]
        alpha = 0.0
        if val > 0:
            alpha = 0.15 + 0.85 * (val / row_max)
        rgba_matrix[i, j, :3] = row_color
        rgba_matrix[i, j, 3] = alpha
        text_color = "black" if alpha < 0.65 else "white"
        ax1.text(j, i, f"{int(val)}", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

ax1.imshow(rgba_matrix, aspect='auto')
ax1.xaxis.tick_top()
ax1.set_xticks(np.arange(N+1))
ax1.set_yticks(np.arange(N))
ax1.set_xticklabels(heatmap_cols, rotation=45, ha='left')
ax1.set_yticklabels(sectors)
ax1.grid(False) 
plt.tight_layout()

# -- Figure 2: Technical Matrix (A) --
fig2, ax2 = plt.subplots(figsize=(12, 8))
plt.title("Technical Matrix A: Normalized Provisioning Recipes", fontsize=16, pad=40)
sns.heatmap(A, annot=True, fmt=".3f", cmap="Greys", 
            xticklabels=sectors, yticklabels=sectors,
            cbar_kws={'label': 'Coefficient (Input per 1 Unit of Output)'}, ax=ax2)

ax2.xaxis.tick_top()
ax2.set_xticklabels(sectors, rotation=45, ha='left')
plt.tight_layout()

# -- Figure 3: Structural Topology & Exchange Ratios (Unified) --
fig3 = plt.figure(figsize=(20, 10))

# Subplot A: Network Topology
ax_top = fig3.add_subplot(121)
ax_top.set_title("Structural Topology: Inter-Sector Dependencies", fontsize=16, pad=20)
G = nx.from_numpy_array(oikos, create_using=nx.DiGraph)

pos = nx.spring_layout(G, k=0.9, seed=42)
edges = G.edges()
weights = [np.log1p(G[u][v]['weight']) * 0.5 for u, v in edges]

scaled_sizes = [prices[i] * 3000 for i in range(N)]
nx.draw_networkx_nodes(G, pos, node_color=hex_colors, node_size=scaled_sizes, edgecolors='black', ax=ax_top)
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrowsize=14, alpha=0.4, connectionstyle="arc3,rad=0.1", ax=ax_top)

# Strip prefix numbers and trailing units for clean node labels
node_labels = {i: re.sub(r' \([^)]*\)', '', sectors[i].split('. ')[1]) for i in range(N)}
nx.draw_networkx_labels(G, pos, node_labels, font_size=9, font_weight="bold", font_color="black", ax=ax_top)
ax_top.axis('off')

# Subplot B: Radar Chart
ax_rad = fig3.add_subplot(122, polar=True)
ax_rad.set_title("Dominant Eigenvector: Absolute Prices & Exchange Ratios", fontsize=16, pad=40)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  
prices_loop = prices.tolist() + [prices[0]]  

ax_rad.plot(angles, prices_loop, color='gray', linewidth=2, linestyle='solid', zorder=1)
ax_rad.fill(angles, prices_loop, color='lightgray', alpha=0.3, zorder=0)

for i in range(N):
    is_num = (i == numeraire_index)
    size = 15 if is_num else 10
    edge = 'black' if is_num else 'gray'
    edge_width = 2.5 if is_num else 1
    
    ax_rad.plot(angles[i], prices[i], marker='o', markersize=size, color=hex_colors[i], 
            markeredgecolor=edge, markeredgewidth=edge_width, zorder=2)
    
    ax_rad.text(angles[i], prices[i] + (max(prices)*0.08), f"{prices[i]:.2f}", 
            ha='center', va='center', fontsize=10, fontweight='bold' if is_num else 'normal')

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels([node_labels[i] for i in range(N)], fontsize=10)

for i, label in enumerate(ax_rad.get_xticklabels()):
    label.set_color(hex_colors[i])
    if i == numeraire_index:
        label.set_fontweight('bold')

plt.tight_layout()
plt.show()

# =====================================================================
# THEORETICAL FOUNDATIONS & FUTURE RESEARCH EXTENSIONS
# =====================================================================
"""
NOTE 1: THE GOVERNANCE OVERLAY MATRIX (Gamma) & WILLIAMSON
- Future ABM steps: Create a second matrix `gamma` of same dimensions.
- Populate it with institutional states: 'M' (Market), 'F' (Firm), 'S' (State), 'C' (Communal).
- Algorithm idea: If an `oikos` block shows high internal density but low external edges 
  (high asset specificity), transaction cost logic should automatically force its `gamma` 
  state from 'M' to 'F' (corporate consolidation).

NOTE 2: N-K DYNAMICS (Stuart Kauffman)
- N = Matrix dimension. K = Density of the `oikos` matrix.
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
- Next step: Introduce a technology shock. Grain output jumps, inputs remain identical.
- The Eigenvalue drops below 1 (the system produces more than it consumes). 
- How is the surplus distributed?
    a) RECIPROCITY: Absorb it into Social Reproduction.
    b) REDISTRIBUTION: Absorb it into Authority to build Non-Basics.
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