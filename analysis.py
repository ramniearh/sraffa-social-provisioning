import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns

# =====================================================================
# CONFIGURATION & PRESETS (Matched to HTML Sandbox)
# =====================================================================
ACTIVE_MODEL = 'capitalist'  # Options: 'diy', 'sraffa_ch1', 'sraffa_ch2', 'sahlins', 'hudson', 'capitalist'

models = {
    'diy': {
        "sectors": ["Wheat", "Iron", "Luxury", "Workers", "Authority/Infra", "Waste"],
        "units": ["tons", "kg", "units", "hours", "infra", "tons"],
        "Z": [
            [100, 50, 10, 250, 50, 10],
            [50, 100, 20, 20, 10, 10],
            [0, 0, 0, 10, 30, 0],
            [300, 150, 20, 1200, 30, 0],
            [20, 20, 10, 40, 10, 0],
            [0, 0, 0, 0, 0, 0]
        ],
        "q": [500, 200, 50, 1800, 100, 20],
        "numeraire": 4, "social": [3, 4, 5],
        "inst": [[1,1,1,1,2,0], [1,1,1,1,2,0], [1,1,1,1,2,0], [1,1,1,3,2,0], [2,2,2,2,2,0], [0,0,0,0,0,0]]
    },
    'sraffa_ch1': {
        "sectors": ["Wheat", "Iron"], "units": ["qr", "tons"],
        "Z": [[280, 120], [12, 8]], "q": [400, 20],
        "numeraire": 0, "social": [],
        "inst": [[1,1], [1,1]]
    },
    'sraffa_ch2': {
        "sectors": ["Wheat", "Iron", "Workers", "Capitalists"], "units": ["qr", "tons", "labor-hrs", "claims"],
        "Z": [[280, 120, 100, 75], [12, 8, 0, 0], [350, 150, 0, 0], [60, 40, 0, 0]],
        "q": [575, 20, 500, 100],
        "numeraire": 0, "social": [2, 3],
        "inst": [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]
    },
    'sahlins': {
        "sectors": ["Foraging", "Biomass", "Tools", "Shelter", "Worker Community", "Feasts"],
        "units": ["kcal", "kg", "units", "days", "hours", "events"],
        "Z": [
            [100, 0, 20, 0, 600, 280], [0, 100, 40, 60, 400, 400], [10, 20, 10, 10, 40, 10],
            [0, 50, 0, 10, 40, 0], [400, 200, 100, 100, 1000, 400], [0, 0, 0, 0, 10, 0]
        ],
        "q": [1000, 1000, 100, 100, 2200, 10],
        "numeraire": 4, "social": [4, 5],
        "inst": np.full((6, 6), 3).tolist()
    },
    'hudson': {
        "sectors": ["Grain", "Materials", "Workers", "Temple / Palace", "Silver / Trade"],
        "units": ["sila", "tons", "months", "months", "shekels"],
        "Z": [
            [100, 50, 600, 250, 0], [50, 50, 200, 190, 10], [400, 200, 50, 300, 50],
            [50, 20, 20, 50, 60], [0, 0, 0, 50, 0]
        ],
        "q": [1000, 500, 1000, 200, 100],
        "numeraire": 3, "social": [2, 3],
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
        "q": [1800, 2400, 260, 100, 9200, 500, 700, 150],
        "numeraire": 6, "social": [4, 5, 6, 7],
        "inst": [[1,1,1,1,1,1,2,0], [1,1,1,1,1,1,2,0], [1,1,1,1,1,1,2,0], [1,1,1,1,1,1,2,0],
                 [1,1,1,1,3,1,2,0], [1,1,1,1,1,1,2,0], [2,2,2,2,2,2,2,0], [0,0,0,0,0,0,0,0]]
    }
}

# =====================================================================
# MATHEMATICAL ENGINE
# =====================================================================
def get_dominant_eigen(M):
    """Rigorous extraction of Perron-Frobenius eigenvalue and left eigenvector."""
    if M.size == 0: return 0.0, np.array([])
    vals, vecs = la.eig(M, left=True, right=False)
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
p = p_raw / p_raw[cfg['numeraire']] if p_raw[cfg['numeraire']] > 0 else p_raw
gross_scale = p * q

# 4. Social Sector Net Flows
net_flows = []
for s in social_idx:
    absorbed = sum(Z[i, s] * p[i] for i in range(N))
    contributed = sum(Z[s, i] * p[s] for i in range(N))
    net_flows.append(absorbed - contributed)

# 5. Institutional & TCE Diagnostics
inst_volumes = np.zeros(4)
for i in range(N):
    for j in range(N):
        inst_volumes[inst[i, j]] += Z[i, j] * p[i]

print(f"=== SYSTEM REPORT: {ACTIVE_MODEL.upper()} ===")
print(f"λ (Economic Core) = {lambda_core:.4f}")
print(f"σ (Full System)   = {sigma_full:.4f}")
print("Non-Basic Processes:", [cfg['sectors'][i] for i in detect_non_basics(Z)])

print("\n--- TRANSACTION COST ECONOMICS (TCE) DIAGNOSTIC ---")
# Calculate input concentration (HHI) for each sector to predict institutional form
for j in range(N):
    col_sum = np.sum(Z[:, j])
    if col_sum == 0: continue
    shares = (Z[:, j] / col_sum) * 100
    hhi = np.sum(shares**2)
    # Most prevalent actual institution in this sector's inputs
    dom_inst = int(np.bincount(inst[:, j][Z[:, j] > 0]).argmax()) if np.any(Z[:, j] > 0) else 0
    tce_pred = "Redistribution/Firm" if hhi > 2500 else "Market"
    actual_inst = ["Unspec", "Market", "Redistribution", "Reciprocity"][dom_inst]
    print(f"{cfg['sectors'][j]:<20} | HHI: {hhi:>4.0f} | TCE Expects: {tce_pred:<19} | Actual Form: {actual_inst}")


# =====================================================================
# PLOTTING ENGINE
# =====================================================================
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f"Provisioning Sandbox Analytical Profile: {ACTIVE_MODEL.upper()}", fontsize=18, fontweight='bold')
gs = fig.add_gridspec(2, 3)

colors_econ = '#81c784'
colors_soc = '#ffab91'
bar_colors = [colors_soc if i in social_idx else colors_econ for i in range(N)]

# --- 1. MULTIPLEX NETWORK (Polanyian Layers) ---
ax_net = fig.add_subplot(gs[0, 0:2])
ax_net.set_title("Multiplex Network: Physical Flows by Institutional Layer", fontsize=14)
G = nx.MultiDiGraph()
inst_colors = {0: '#b0bec5', 1: '#388e3c', 2: '#f9a825', 3: '#e64a19'}
for i in range(N):
    G.add_node(i, label=cfg['sectors'][i], size=max(10, np.sqrt(gross_scale[i])*2))
for i in range(N):
    for j in range(N):
        if Z[i, j] > 0:
            layer = inst[i, j]
            G.add_edge(i, j, weight=Z[i, j]*p[i], color=inst_colors[layer])

pos = nx.spring_layout(G, k=1.5, seed=42)
node_sizes = [G.nodes[n]['size']*20 for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=bar_colors, node_size=node_sizes, edgecolors='black', ax=ax_net)

edges = G.edges(data=True)
edge_colors = [d['color'] for u, v, d in edges]
edge_weights = [np.log1p(d['weight']) * 0.5 for u, v, d in edges]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, arrowsize=12, connectionstyle="arc3,rad=0.1", ax=ax_net)
nx.draw_networkx_labels(G, pos, labels={i: cfg['sectors'][i] for i in range(N)}, font_size=9, font_weight='bold', ax=ax_net)

# Custom legend for network
patches = [mpatches.Patch(color=v, label=k) for k, v in {"Market": '#388e3c', "Redistribution": '#f9a825', "Reciprocity": '#e64a19'}.items()]
ax_net.legend(handles=patches, loc='upper left')

# --- 2. GROSS SECTORAL SCALE ---
ax_scale = fig.add_subplot(gs[1, 0])
ax_scale.set_title("Gross Sectoral Scale (in Numeraire)", fontsize=12)
ax_scale.barh(cfg['sectors'], gross_scale, color=bar_colors, edgecolor='black')
ax_scale.invert_yaxis()

# --- 3. SOCIAL SECTOR NET FLOWS ---
ax_flow = fig.add_subplot(gs[1, 1])
ax_flow.set_title("Net Structural Position (Social Sectors)", fontsize=12)
flow_labels = [cfg['sectors'][i] for i in social_idx]
flow_colors = ['#c5e1a5' if val < 0 else '#ef9a9a' for val in net_flows] # Green if contributing, Red if absorbing
ax_flow.barh(flow_labels, net_flows, color=flow_colors, edgecolor='black')
ax_flow.axvline(0, color='black', linewidth=1)
ax_flow.invert_yaxis()

# --- 4. INSTITUTIONAL VOLUME (Doughnut) ---
ax_inst = fig.add_subplot(gs[1, 2])
ax_inst.set_title("Institutional Integration Breakdown", fontsize=12)
labels_inst = ['Unspecified', 'Market', 'Redistribution', 'Reciprocity']
colors_doughnut = ['#eceff1', '#81c784', '#ffd54f', '#ff8a65']
wedges, texts, autotexts = ax_inst.pie(inst_volumes, labels=labels_inst, colors=colors_doughnut, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
for text in autotexts: text.set_fontsize(9)

# --- 5. PROVISIONING CENTRALITY (Radar) ---
ax_radar = fig.add_subplot(gs[0, 2], polar=True)
ax_radar.set_title("Provisioning Centrality (Left Eigenvector)", y=1.1, fontsize=12)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
p_loop = p.tolist() + [p[0]]
ax_radar.plot(angles, p_loop, color='#2e7d32', linewidth=2)
ax_radar.fill(angles, p_loop, color='#2e7d32', alpha=0.2)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(cfg['sectors'], fontsize=8)

plt.tight_layout()
plt.show()