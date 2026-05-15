import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

# =====================================================================
# CONFIGURATION & PRESETS (Matched exactly to HTML Sandbox)
# =====================================================================
# Select preset: 'diy', 'sraffa_ch1', 'sraffa_ch2', 'sahlins', 'hudson', 'capitalist'
ACTIVE_MODEL = 'capitalist'

# Institutional Annotation Codes:
# 0: Unspecified, 1: Market Exchange, 2: Redistribution, 3: Reciprocity, 4: Market (Organizations)
INST_LABELS = ['Unspecified', 'Market Exchange', 'Redistribution', 'Reciprocity', 'Market (Organizations)']
INST_COLORS = ['#eceff1', '#81c784', '#ffd54f', '#ff8a65', '#26a69a'] # 4 is Teal

models = {
    'diy': {
        "sectors": ["Wheat", "Iron", "Luxury", "Workers", "Authority/Infra", "Waste"], "units": ["tons", "kg", "units", "hours", "infra", "tons"],
        "Z": [
            [100, 50, 10, 250, 50, 10], [50, 100, 20, 20, 10, 10], [0, 0, 0, 10, 30, 0],
            [300, 150, 20, 1200, 30, 0], [20, 20, 10, 40, 10, 0], [0, 0, 0, 0, 0, 0]
        ],
        "q": [500, 200, 50, 1800, 100, 20], "numeraire": 4, "social": [3, 4, 5],
        "inst": [[1,1,1,1,2,0], [1,1,1,1,2,0], [1,1,1,1,2,0], [1,1,1,3,2,0], [2,2,2,2,2,0], [0,0,0,0,0,0]]
    },
    'sraffa_ch1': {
        "sectors": ["Wheat", "Iron"], "units": ["qr", "tons"],
        "Z": [[280, 120], [12, 8]], "q": [400, 20], "numeraire": 0, "social": [],
        "inst": [[1,1], [1,1]]
    },
    'sraffa_ch2': {
        "sectors": ["Wheat", "Iron", "Workers", "Capitalists"], "units": ["qr", "tons", "labor-hrs", "claims"],
        "Z": [[280, 120, 100, 75], [12, 8, 0, 0], [350, 150, 0, 0], [60, 40, 0, 0]],
        "q": [575, 20, 500, 100], "numeraire": 0, "social": [2, 3],
        "inst": [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]
    },
    'sahlins': {
        "sectors": ["Foraging", "Biomass", "Tools", "Shelter", "Worker Community", "Feasts"], "units": ["kcal", "kg", "units", "days", "hours", "events"],
        "Z": [
            [100, 0, 20, 0, 600, 280], [0, 100, 40, 60, 400, 400], [10, 20, 10, 10, 40, 10],
            [0, 50, 0, 10, 40, 0], [400, 200, 100, 100, 1000, 400], [0, 0, 0, 0, 10, 0]
        ],
        "q": [1000, 1000, 100, 100, 2200, 10], "numeraire": 4, "social": [4, 5],
        "inst": np.full((6, 6), 3).tolist()
    },
    'hudson': {
        "sectors": ["Grain", "Materials", "Workers", "Temple / Palace", "Silver / Trade"], "units": ["sila", "tons", "months", "months", "shekels"],
        "Z": [
            [100, 50, 600, 250, 0], [50, 50, 200, 190, 10], [400, 200, 50, 300, 50],
            [50, 20, 20, 50, 60], [0, 0, 0, 50, 0]
        ],
        "q": [1000, 500, 1000, 200, 100], "numeraire": 3, "social": [2, 3],
        "inst": [[2,2,2,2,1], [2,2,2,2,1], [2,2,2,2,1], [2,2,2,2,1], [1,1,1,1,1]]
    },
    'capitalist': {
        "sectors": ["Industry", "Services", "Tech Innovation", "Veblenian / Luxury", "Work (labor+care)", "F.I.R.E.", "State", "Biosphere"],
        "units": ["tons", "MWh", "patents", "units", "hours", "claims", "value", "tons"],
        "Z": [
            [400,300,100,50,600,50,200,100], [300,400,200,50,1000,100,300,50], [50,50,10,0,0,100,50,0],
            [0,0,0,0,0,100,0,0], [800,1000,100,100,6900,100,200,0], [100,100,50,50,50,50,100,0],
            [100,100,50,50,200,100,100,0], [200,50,0,0,0,0,0,0]
        ],
        "q": [1800, 2400, 260, 100, 9200, 500, 700, 150], "numeraire": 6, "social": [4, 5, 6, 7],
        # Modified to include '4' (Market Organizations) for highly concentrated corporate flows
        "inst": [
            [4,1,1,1,1,4,2,0], [1,4,1,1,1,4,2,0], [4,4,4,1,1,4,2,0], [1,1,1,1,1,1,2,0],
            [1,1,1,1,3,1,2,0], [4,4,4,4,4,4,2,0], [2,2,2,2,2,2,2,0], [0,0,0,0,0,0,0,0]
        ]
    }
}

# =====================================================================
# MATHEMATICAL ENGINE
# =====================================================================
def get_dominant_eigen(M):
    """Rigorous extraction of Perron-Frobenius eigenvalue and left eigenvector."""
    if M.shape[0] == 0: return 0.0, np.array([])
    vals, vecs = la.eig(M, left=True, right=False)
    # Find the dominant real eigenvalue
    idx = np.argmax(np.real(vals))
    return np.real(vals[idx]), np.abs(np.real(vecs[:, idx]))

def detect_non_basics(Z):
    N = Z.shape[0]
    G = nx.from_numpy_array(Z, create_using=nx.DiGraph)
    non_basics = []
    for i in range(N):
        reachable = nx.descendants(G, i) | {i}
        if len(reachable) < N: non_basics.append(i)
    return non_basics

# Load data
cfg = models[ACTIVE_MODEL]
Z = np.array(cfg['Z'], dtype=float)
q = np.array(cfg['q'], dtype=float)
N = len(cfg['sectors'])
social_idx = cfg['social']
econ_idx = [i for i in range(N) if i not in social_idx]
inst = np.array(cfg['inst'])

# 1. Technical Matrix (A)
A = np.zeros_like(Z)
for j in range(N):
    if q[j] > 0: A[:, j] = Z[:, j] / q[j]

# 2. Reproduction Values (Eigenvalues)
A_econ = A[np.ix_(econ_idx, econ_idx)]
lambda_core, _ = get_dominant_eigen(A_econ)
sigma_full, p_raw = get_dominant_eigen(A)

# 3. Provisioning Centrality
numeraire = cfg['numeraire']
p = p_raw / p_raw[numeraire] if p_raw[numeraire] > 0 else p_raw
gross_scale = p * q

# 4. Diagnostics
non_basics = detect_non_basics(Z)
row_sums = np.sum(Z, axis=1)
balances = q - row_sums


# =====================================================================
# TERMINAL REPORT (Matching HTML "System Reproduction Parameters")
# =====================================================================
print("="*80)
print(f" SYSTEM REPRODUCTION PARAMETERS: {ACTIVE_MODEL.upper()}")
print("="*80)

print("\n--- Viability, Reproduction, and Sectoral Balances ---")
print(f"λ (Economic Core) = {lambda_core:.4f}")
if lambda_core < 0.99: print("   -> Economic sub-system generates a surplus.")
elif lambda_core > 1.01: print("   -> Economic sub-system operates at a physical deficit.")
else: print("   -> Economic sub-system is at exact subsistence.")

print(f"\nσ (Full System)   = {sigma_full:.4f}")
if sigma_full < 0.99: print("   -> System-level physical surplus.")
elif sigma_full > 1.01: print("   -> System-level physical deficit.")
else: print("   -> System at exact reproduction.")

print("\nBalance Table:")
is_balanced = True
for i in range(N):
    if abs(balances[i]) > 0.01:
        is_balanced = False
        status = "Unallocated Surplus" if balances[i] > 0 else "Deficit"
        print(f"   [{cfg['sectors'][i]}] {status} of {abs(balances[i]):.2f} {cfg['units'][i]}")
if is_balanced: print("   -> The matrix is closed. All outputs are absorbed as inputs.")

print("\nStructural Topology:")
if non_basics:
    print(f"   Non-Basic Processes detected: {[cfg['sectors'][i] for i in non_basics]}")
else:
    print("   All processes are structurally Basic.")

print(f"\n--- Provisioning Centrality (Normalized to {cfg['sectors'][numeraire]}) ---")
for i in range(N):
    marker = "*" if i == numeraire else " "
    print(f"{marker} 1 {cfg['units'][i]:<8} {cfg['sectors'][i]:<25}: {p[i]:.4f}")
print("="*80)


# =====================================================================
# PLOTTING ENGINE (Separate Windows)
# =====================================================================
sns.set_theme(style="whitegrid")
node_colors = ['#ffab91' if i in social_idx else '#a5d6a7' for i in range(N)]

# 1. Technical Dependency Network (A) - Intensive
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.set_title("Technical Dependency Network (A Matrix)\nNode = Per-Unit Centrality | Edge = Coefficient", fontsize=14)
GA = nx.DiGraph()
for i in range(N): GA.add_node(i, label=cfg['sectors'][i], size=max(100, p[i]*1500))
for i in range(N):
    for j in range(N):
        if A[i, j] > 0: GA.add_edge(i, j, weight=A[i, j]*5, color=INST_COLORS[inst[i, j]])
pos = nx.spring_layout(GA, k=1.5, seed=42)
nx.draw_networkx_nodes(GA, pos, node_color=node_colors, node_size=[GA.nodes[n]['size'] for n in GA.nodes()], edgecolors='black', ax=ax1)
edges = GA.edges(data=True)
nx.draw_networkx_edges(GA, pos, edge_color=[d['color'] for u,v,d in edges], width=[d['weight'] for u,v,d in edges], arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax1)
nx.draw_networkx_labels(GA, pos, labels={i: cfg['sectors'][i] for i in range(N)}, font_size=10, font_weight='bold', ax=ax1)

# 2. Physical Flow Network (Z) - Extensive
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.set_title("Physical Flow Network (Z Matrix)\nNode = Gross Sectoral Scale | Edge = Absolute Flow Vol", fontsize=14)
GZ = nx.DiGraph()
for i in range(N): GZ.add_node(i, label=cfg['sectors'][i], size=max(100, np.sqrt(gross_scale[i])*100))
for i in range(N):
    for j in range(N):
        if Z[i, j] > 0: GZ.add_edge(i, j, weight=np.log1p(Z[i, j] * p[i])*1.5, color=INST_COLORS[inst[i, j]])
nx.draw_networkx_nodes(GZ, pos, node_color=node_colors, node_size=[GZ.nodes[n]['size'] for n in GZ.nodes()], edgecolors='black', ax=ax2)
edges_z = GZ.edges(data=True)
nx.draw_networkx_edges(GZ, pos, edge_color=[d['color'] for u,v,d in edges_z], width=[d['weight'] for u,v,d in edges_z], arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax2)
nx.draw_networkx_labels(GZ, pos, labels={i: cfg['sectors'][i] for i in range(N)}, font_size=10, font_weight='bold', ax=ax2)

# 3. Gross Sectoral Scale (Bar Chart)
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_title("Gross Sectoral Scale (in Numeraire)", fontsize=14)
ax3.barh(cfg['sectors'], gross_scale, color=node_colors, edgecolor='black')
ax3.invert_yaxis()

# 4. Gross Input Absorption by Social Sector (Stacked Horizontal)
fig4, ax4 = plt.subplots(figsize=(10, 3))
ax4.set_title("Gross Input Absorption (% of Total Systemic Scale)", fontsize=14)
total_scale = np.sum(gross_scale)
soc_absorption = []
for s in social_idx:
    absorbed = sum(Z[i, s] * p[i] for i in range(N))
    soc_absorption.append((absorbed / total_scale) * 100 if total_scale > 0 else 0)
remainder = max(0, 100 - sum(soc_absorption))
left = 0
colors_stack = ['#bf360c', '#d84315', '#e64a19', '#ff5722']
for idx, s in enumerate(social_idx):
    if soc_absorption[idx] > 0:
        ax4.barh(['System Output'], [soc_absorption[idx]], left=left, color=colors_stack[idx % len(colors_stack)], edgecolor='white', label=cfg['sectors'][s])
        left += soc_absorption[idx]
if remainder > 0:
    ax4.barh(['System Output'], [remainder], left=left, color='#cfd8dc', edgecolor='white', label='Intermediate Econ Production')
ax4.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3)

# 5. Net Flow Position of Social Sectors (Bar)
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.set_title("Net Flow Position of Social Sectors\n(Inputs Received vs Outputs Contributed)", fontsize=14)
net_flows = []
for s in social_idx:
    absorbed = sum(Z[i, s] * p[i] for i in range(N))
    contributed = sum(Z[s, i] * p[s] for i in range(N))
    net_flows.append(absorbed - contributed)
flow_colors = ['#ef9a9a' if val >= 0 else '#c5e1a5' for val in net_flows]
ax5.barh([cfg['sectors'][i] for i in social_idx], net_flows, color=flow_colors, edgecolor='black')
ax5.axvline(0, color='black', linewidth=1)
ax5.invert_yaxis()

# 6. Institutional Integration Breakdown (Doughnut)
fig6, ax6 = plt.subplots(figsize=(8, 8))
ax6.set_title("Institutional Integration Breakdown", fontsize=14)
inst_vols = np.zeros(5)
for i in range(N):
    for j in range(N):
        inst_vols[inst[i, j]] += Z[i, j] * p[i]
filtered_vols = []
filtered_labels = []
filtered_colors = []
for i in range(5):
    if inst_vols[i] > 0:
        filtered_vols.append(inst_vols[i])
        filtered_labels.append(INST_LABELS[i])
        filtered_colors.append(INST_COLORS[i])
ax6.pie(filtered_vols, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))

# 7. Provisioning Centrality Profile (Radar)
fig7, ax7 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax7.set_title("Provisioning Centrality Profile", y=1.1, fontsize=14)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
p_loop = p.tolist() + [p[0]]
ax7.plot(angles, p_loop, color='#2e7d32', linewidth=2)
ax7.fill(angles, p_loop, color='#2e7d32', alpha=0.2)
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(cfg['sectors'], fontsize=10)

# 8. Technical Coefficients Matrix (Heatmap)
fig8, ax8 = plt.subplots(figsize=(10, 8))
ax8.set_title("Technical Coefficients Matrix (A)", fontsize=14)
sns.heatmap(A, annot=True, fmt=".3f", cmap="Greys", xticklabels=cfg['sectors'], yticklabels=cfg['sectors'], cbar_kws={'label': 'Units of input per unit of output'}, ax=ax8)
ax8.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')

# Show all generated windows simultaneously
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

NOTE 9: MONEY AS ITS OWN SOCIAL SECTOR SINK, and post-keynesian effective demand. Also versus Leontieff *monetary* matrix.
where regulation school?


NOTE 10: INSTITUTIONAL BLOCK MATRIX AND/OR POLITICAL SETTLEMENTS (e.g. extend sraffa wage-profit logic to mulitple social sectors)
- The current topological matrix evaluates all sectors using a single mathematical logic (Provisioning Centrality).
- Future extension: Fracture the single topology into an interacting Block Matrix, where distinct institutional spheres obey distinct algebraic laws of motion.
- Example formulation: A Capitalist Market governed by a uniform profit rate (r), a State governed by fiat tax extraction (tau), and a Household sector governed by the shadow-price of unpaid reproduction (v). 
- This framework allows researchers to model the "crisis of social reproduction" algebraically—demonstrating, for instance, how a rising capitalist profit rate mathematically forces the implicit value of unpaid care (v) to plummet, requiring households to cannibalize their own social and emotional bandwidth to subsidize the market core.
"""