import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx

# =====================================================================
# SRAFFIAN MATRIX OF PROVISIONING & SOCIAL REPRODUCTION
# Expanding the Sraffian framework to attempt to develop a unified 
# formalism for social and economic provisioning systems. 
#
# This model embeds non-market institutions and human social reproduction 
# directly into the physical input-output matrix, treating distribution 
# as a structural prior to exchange value.
# =====================================================================

# Select the model to run: 'sahlins', 'hudson', 'capitalist', or 'diy'
ACTIVE_MODEL = 'hudson'

models = {}

# ---------------------------------------------------------
# 1. THE SAHLINS MODEL: Hunter-Gatherer Affluence
# ---------------------------------------------------------
models['sahlins'] = {
    "name": "Sahlins: Original Affluent Society",
    "sectors": ["0. Foraging (kcal)", "1. Biomass/Wood (kg)", "2. Stone Tools (units)", "3. Temporary Shelter", "4. Workers & Community (hours)"],
    "colors": ['#DEB887', '#228B22', '#708090', '#CD5C5C', '#2E8B57'],
    "oikos": np.array([
        [  50,   0,   0,   0,  950], # Foraging
        [   0,  50,   0,  50,  400], # Biomass
        [  10,  20,  10,  10,   50], # Tools
        [   0,  50,   0,   0,   50], # Shelter
        [ 940, 380,  90,  40,  550]  # Workers (Highly self-referential leisure/care)
    ], dtype=float),
    "total_output": np.array([1000, 500, 100, 100, 2000], dtype=float),
    "numeraire_index": 4, 
    "social_indices": [4],
    "surplus_goods_indices": [] # In Sahlins, the concept of a dedicated sterile surplus good does not exist
}

# ---------------------------------------------------------
# 2. THE HUDSON MODEL: Mesopotamian Palace Economy
# ---------------------------------------------------------
models['hudson'] = {
    "name": "Hudson: Mesopotamian Bronze Age",
    "sectors": [
        "0. Grain (tons)", "1. Energy (MWh)", "2. Raw Materials (tons)", 
        "3. Tools & Implements", "4. Food Rations", "5. Shelter & Housing",  
        "6. Veblenian/Surplus Goods", "7. Workers & Reproduction", "8. Authority & Temple"   
    ],
    "colors": ['#DEB887', '#FFD700', '#B0C4DE', '#708090', '#F4A460', '#CD5C5C', '#C71585', '#2E8B57', '#6A5ACD'],
    "oikos": np.array([
        [200, 100,   0,   0,1630,   0,   0,  60,  10], # Grain
        [100, 100, 200, 100, 250, 100,  50,  80,  20], # Energy
        [ 50, 100,  50, 150,   0,  50, 100,   0,   0], # Raw Materials
        [ 50,  20,  30,  20,  40,  40,   0,  60,  40], # Tools
        [100,  50, 100, 100, 150, 100,   0, 800, 100], # Food Rations
        [ 50,  50,  50,  50,  80,  20,   0, 170,  30], # Shelter
        [  0,   0,   0,   0,   0,   0,   0,   0, 100], # Veblenian/Surplus Goods
        [500, 300, 400, 200, 550, 300, 150, 400, 250], # Workers & Reproduction
        [ 30,  40,  50,  30,  50,  30,   0,  40,  30]  # Authority
    ], dtype=float),
    "total_output": np.array([2000, 1000, 500, 300, 1500, 500, 100, 3050, 300], dtype=float),
    "numeraire_index": 8,
    "social_indices": [7, 8],
    "surplus_goods_indices": [6] # Veblenian goods represent explicit, sterile surplus extraction
}

# ---------------------------------------------------------
# 3. THE CAPITALIST MODEL: Financialization
# ---------------------------------------------------------
models['capitalist'] = {
    "name": "Financialized Capitalism",
    "sectors": [
        "0. Primary (Ag/Extract)", "1. Secondary (Mfg)", "2. Tertiary (Services)", 
        "3. F.I.R.E. (Rent/Debt)", "4. Working Class", "5. Capitalist Class", "6. The State"         
    ],
    "colors": ['#8FBC8F', '#708090', '#87CEFA', '#FF4500', '#4682B4', '#FFD700', '#696969'],
    "oikos": np.array([
        [ 200,  800,  400,    0,  500,   50,   50], # Primary
        [ 400,  500,  600,  100,  600,  100,  200], # Secondary
        [ 200,  300,  400,  300, 1200,  300,  300], # Services
        [ 200,  300,  300,  200,  400,   50,   50], # F.I.R.E.
        [ 800, 1200, 1400,  400,    0,    0,  200], # Working Class
        [ 150,  200,  100,   50,    0,    0,    0], # Capitalists
        [  50,  200,  200,   50,  300,  100,  100]  # The State
    ], dtype=float),
    "total_output": np.array([2000, 2500, 3000, 1500, 4000, 500, 1000], dtype=float),
    "numeraire_index": 4, 
    "social_indices": [4, 5, 6],
    "surplus_goods_indices": [3, 5] # FIRE and pure Capitalist Property Rights act as the surplus extraction layers
}

# ---------------------------------------------------------
# 4. DIY CONFIGURATION
# ---------------------------------------------------------
models['diy'] = {
    "name": "Custom User Matrix",
    "sectors": ["0. Wheat", "1. Iron", "2. Veblenian/Surplus", "3. Workers", "4. Capitalists"],
    "colors": ['#DEB887', '#B0C4DE', '#C71585', '#2E8B57', '#FFD700'],
    "oikos": np.array([
        [100,  50,   0, 250, 100], 
        [ 50, 100,  10,  20,  20],
        [  0,   0,   0,   0,  50], 
        [300,  50,  40,  50,  60], 
        [ 50,   0,   0,   0,   0]  
    ], dtype=float),
    "total_output": np.array([500, 200, 50, 500, 50], dtype=float),
    "numeraire_index": 3,
    "social_indices": [3, 4],
    "surplus_goods_indices": [2]
}

# Load active model
cfg = models[ACTIVE_MODEL]
sectors = cfg["sectors"]
N = len(sectors)
oikos = cfg["oikos"]
total_output = cfg["total_output"]
hex_colors = cfg["colors"]
numeraire_index = cfg["numeraire_index"]
social_indices = cfg["social_indices"]
surplus_goods_indices = cfg["surplus_goods_indices"]

# FIX: Translate hex codes to RGB arrays for the Heatmap visualization
custom_colors = [mcolors.to_rgb(c) for c in hex_colors]

oikos_with_total = np.column_stack((oikos, total_output))
heatmap_cols = sectors + ["TOTAL OUTPUT"]

# =====================================================================
# MODULE 1: MATHEMATICAL DIAGNOSTICS & SUBSISTENCE CHECK
# =====================================================================
print("\n" + "="*70)
print(f" LOADING MODEL: {cfg['name'].upper()}")
print("="*70)

# 1. Physical Balance
row_sums = np.sum(oikos, axis=1)
net_surplus = total_output - row_sums

print("\n--- 1. PHYSICAL BALANCE ---")
is_balanced = True
for i in range(N):
    if net_surplus[i] > 1e-5:
        print(f"  [{sectors[i]}] SURPLUS: +{net_surplus[i]:.2f} units")
        is_balanced = False
    elif net_surplus[i] < -1e-5:
        print(f"  [{sectors[i]}] DEFICIT: {net_surplus[i]:.2f} units")
        is_balanced = False
        
if is_balanced: print("  All sectors balanced (Perfect Subsistence).")

# 2. Linear Algebra (Eigenvector & Prices)
A = np.zeros((N, N))
for j in range(N):
    if total_output[j] > 0: A[:, j] = oikos[:, j] / total_output[j]

eigenvalues, eigenvectors = np.linalg.eig(A.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
dom_lambda = np.real(eigenvalues[idx])

if not np.isclose(dom_lambda, 1.0, atol=1e-4):
    print(f"\n>>> SYSTEM HALT: Matrix is not at Subsistence (λ = {dom_lambda:.4f}). <<<")
    print(">>> Please adjust the data to ensure Total Output equals the Row Sums.")
    sys.exit() 

p_raw = np.abs(np.real(eigenvectors[:, idx]))
prices = p_raw / p_raw[numeraire_index]


# =====================================================================
# MODULE 2: RAW DATA EXPORT (TERMINAL)
# =====================================================================
print("\n--- 2. THE MATRIX (Raw Physical Units) ---")
header = f"{'Sector':<35}" + "".join([f"{i:>6}" for i in range(N)]) + " | Total"
print(header)
for i in range(N):
    row_str = f"{sectors[i][:33]:<35}" + "".join([f"{int(x):>6}" for x in oikos[i]]) + f" | {int(total_output[i]):>5}"
    print(row_str)

print("\n--- 3. ABSOLUTE PRICES ---")
for i in range(N):
    print(f"{sectors[i]:<35}: {prices[i]:>8.4f} PDUs")


# =====================================================================
# MODULE 3: TOTAL VALUE VS SURPLUS APPROPRIATION
# =====================================================================
print("\n--- 4. STRUCTURAL VALUE APPROPRIATION (BASIC VS SURPLUS) ---")

social_basic_values = []
social_surplus_values = []
social_labels = []
social_bar_colors = []

total_social_value = 0.0

for s_idx in social_indices:
    # Calculate Basic Value (Goods required for provisioning)
    basic_val = sum(oikos[i, s_idx] * prices[i] for i in range(N) if i not in surplus_goods_indices)
    
    # Calculate Surplus Value (Explicitly non-basic Veblenian/Extraction goods)
    surplus_val = sum(oikos[i, s_idx] * prices[i] for i in surplus_goods_indices)
    
    total_val = basic_val + surplus_val
    
    social_basic_values.append(basic_val)
    social_surplus_values.append(surplus_val)
    social_labels.append(re.sub(r' \([^)]*\)', '', sectors[s_idx].split('. ')[1]))
    social_bar_colors.append(hex_colors[s_idx])
    
    total_social_value += total_val

for i, label in enumerate(social_labels):
    class_total = social_basic_values[i] + social_surplus_values[i]
    share = (class_total / total_social_value) * 100
    surplus_ratio = (social_surplus_values[i] / class_total) * 100 if class_total > 0 else 0
    print(f"  {label:<30}: {share:>5.1f}% of System Total (of which {surplus_ratio:.1f}% is pure surplus)")
print("="*70 + "\n")


# =====================================================================
# MODULE 4: VISUALIZATIONS
# =====================================================================
sns.set_theme(style="whitegrid")

# -- Figure 1: Physical Matrix Flows --
fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.title(f"{cfg['name']}\nPhysical Matrix: Aggregate Intra-System Flows", fontsize=16, pad=30)
rgba_matrix = np.ones((N, N+1, 4)) 
for i in range(N):
    row_color = np.array(custom_colors[i])
    row_max = np.max(oikos_with_total[i])
    for j in range(N+1):
        val = oikos_with_total[i, j]
        alpha = 0.0
        if val > 0: alpha = 0.15 + 0.85 * (val / row_max)
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
fig2, ax2 = plt.subplots(figsize=(10, 8))
plt.title("Technical Matrix A: Normalized Provisioning Recipes", fontsize=16, pad=30)
sns.heatmap(A, annot=True, fmt=".3f", cmap="Greys", xticklabels=sectors, yticklabels=sectors, cbar_kws={'label': 'Coefficient (Input per 1 Unit of Output)'}, ax=ax2)
ax2.xaxis.tick_top()
ax2.set_xticklabels(sectors, rotation=45, ha='left')
plt.tight_layout()

# -- Figure 3: The 3-Panel Dashboard --
fig3 = plt.figure(figsize=(24, 8))
plt.suptitle(f"System Topology & Value Distribution: {cfg['name']}", fontsize=18, fontweight='bold', y=1.05)

# Use GridSpec to shrink the bar chart panel (ratios: 1.5, 1.5, 0.8)
gs = fig3.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 0.9])

# Subplot 3.1: Network Topology
ax_top = fig3.add_subplot(gs[0, 0])
ax_top.set_title("Structural Topology", fontsize=14, pad=15)
G = nx.from_numpy_array(oikos, create_using=nx.DiGraph)
pos = nx.spring_layout(G, k=0.9, seed=42)
weights = [np.log1p(G[u][v]['weight']) * 0.5 for u, v in G.edges()]
scaled_sizes = [prices[i] * 3000 for i in range(N)] 

nx.draw_networkx_nodes(G, pos, node_color=hex_colors, node_size=scaled_sizes, edgecolors='black', ax=ax_top)
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrowsize=10, alpha=0.3, connectionstyle="arc3,rad=0.1", ax=ax_top)
node_labels = {i: re.sub(r' \([^)]*\)', '', sectors[i].split('. ')[1]) for i in range(N)}
nx.draw_networkx_labels(G, pos, node_labels, font_size=9, font_weight="bold", font_color="black", ax=ax_top)
ax_top.axis('off')

# Subplot 3.2: Radar Chart (Prices)
ax_rad = fig3.add_subplot(gs[0, 1], polar=True)
ax_rad.set_title(f"Dominant Eigenvector (Relative Exchange Ratios)\nNumeraire: {node_labels[numeraire_index]}", fontsize=14, pad=20)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  
prices_loop = prices.tolist() + [prices[0]]  

ax_rad.plot(angles, prices_loop, color='gray', linewidth=2, linestyle='solid', zorder=1)
ax_rad.fill(angles, prices_loop, color='lightgray', alpha=0.3, zorder=0)

for i in range(N):
    is_num = (i == numeraire_index)
    size = 12 if is_num else 8
    ax_rad.plot(angles[i], prices[i], marker='o', markersize=size, color=hex_colors[i], markeredgecolor='black', markeredgewidth=1.5 if is_num else 0.5, zorder=2)
    ax_rad.text(angles[i], prices[i] + (max(prices)*0.08), f"{prices[i]:.2f}", ha='center', va='center', fontsize=9, fontweight='bold' if is_num else 'normal')

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels([node_labels[i] for i in range(N)], fontsize=9)

# Subplot 3.3: STACKED Bar Chart (Basic vs Surplus Appropriation)
ax_bar = fig3.add_subplot(gs[0, 2])
ax_bar.set_title("Value Appropriation\n(Basic vs. Surplus Goods)", fontsize=14, pad=15)

# Convert to percentages of the total pie
basic_pcts = [(v / total_social_value) * 100 for v in social_basic_values]
surplus_pcts = [(v / total_social_value) * 100 for v in social_surplus_values]

bar_width = 0.55
bars_basic = ax_bar.bar(social_labels, basic_pcts, color=social_bar_colors, edgecolor='black', width=bar_width, alpha=0.6, label="Basic/Provisioning")
bars_surplus = ax_bar.bar(social_labels, surplus_pcts, bottom=basic_pcts, color=social_bar_colors, edgecolor='black', width=bar_width, hatch='//', label="Pure Surplus/Veblenian")

ax_bar.set_ylabel("Share of Systemic Bandwidth (%)", fontsize=11)
ax_bar.set_ylim(0, 100)
ax_bar.tick_params(axis='x', labelsize=10, rotation=15)

# Add percentage labels on top of the stacked bars
for i in range(len(social_labels)):
    total_y = basic_pcts[i] + surplus_pcts[i]
    ax_bar.text(i, total_y + 1.5, f"{total_y:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Legend for the hatch pattern
import matplotlib.patches as mpatches
basic_patch = mpatches.Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Basic Reproduction')
surplus_patch = mpatches.Patch(facecolor='gray', hatch='//', edgecolor='black', label='Surplus / Veblenian')
ax_bar.legend(handles=[basic_patch, surplus_patch], loc='upper left', fontsize=9)

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
- Anthropology/Gift Economies: Chris Gregory (1982) "Gifts and Commodities". 
- Feminist Economics/Social Reproduction: Antonella Picchio (1992) "Social Reproduction". 
- Institutionalism/Pre-Capitalist Surplus: Sergio Cesaratto (2015/2020). 
- Ecological I-O: Bruce Hannon (1973) "The Structure of Ecosystems". 

NOTE 8: ENTROPY & DISSIPATIVE STRUCTURES (Prigogine)
- In Sraffa's system, surplus is either perfectly consumed or extracted, allowing the cycle to restart.
- Through a thermodynamic lens, the Sraffian matrix is a "Dissipative Structure" operating far from equilibrium.
- A surplus represents an injection of "free energy" (exergy). If the society does not continuously 
  dissipate that energy through expanded consumption, luxury, or warfare (acting as entropy sinks), 
  the system faces a thermodynamic/accumulation crisis and structurally collapses.
"""