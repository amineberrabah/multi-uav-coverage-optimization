# le centroïde de chaque drone est prédit par le CNN
# l'ajout de drone est prédit par le CNN :
    #le CNN observe la scène complète et prédit directement
    #où placer le nouveau drone (n+1) pour maximiser la couverture
#le systeme s'arrete si :
    # if convergence and cout_total <= COUT_SEUIL
    # convergence and n_drones == MAX_DRONES


from shapely import voronoi_polygons, MultiPoint
from shapely.geometry import box, Point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter
import random

np.random.seed(44)
torch.manual_seed(44)
random.seed(44)

# ════════════════════════════════════════════════
#  FONCTION DESSIN DRONE (forme quadrirotor noir)
# ════════════════════════════════════════════════
def draw_drone(ax, x, y, size=3.5):
    """Dessine un drone quadrirotor noir centré en (x, y)."""
    arm   = size * 0.8          # longueur demi-bras
    r_mot = size * 0.35         # rayon moteur
    r_bod = size * 0.28         # rayon corps central

    # Bras en croix (×)
    for angle in [45, 135]:
        dx = arm * np.cos(np.radians(angle))
        dy = arm * np.sin(np.radians(angle))
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color='black', lw=1.5, zorder=6, solid_capstyle='round')

    # Hélices aux extrémités
    for angle in [45, 135, 225, 315]:
        mx = x + arm * np.cos(np.radians(angle))
        my = y + arm * np.sin(np.radians(angle))
        circle = plt.Circle((mx, my), r_mot, color='black', fill=True, zorder=7)
        ax.add_patch(circle)
        # petit cercle blanc intérieur (effet hélice)
        inner = plt.Circle((mx, my), r_mot * 0.45, color='white', fill=True, zorder=8)
        ax.add_patch(inner)

    # Corps central
    body = plt.Circle((x, y), r_bod, color='black', fill=True, zorder=9)
    ax.add_patch(body)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ════════════════════════════════════════════════
#  PARAMÈTRES
# ════════════════════════════════════════════════
R          = 20
n_drones   = 2
n_cibles   = 100
MAX_DRONES = 20

alpha = 0.6
beta  = 0.4

xmin, xmax = 0, 100
ymin, ymax = 0, 100

GRID   = 50
LR     = 1e-3
LR_COV = 1e-3
DEVICE = torch.device("cpu")

# ════════════════════════════════════════════════
#  DONNÉES
# ════════════════════════════════════════════════
centroides       = np.zeros((n_drones, 2))
positions_drones = np.random.uniform([xmin, ymin], [xmax, ymax], (n_drones, 2))
cibles           = np.random.uniform([xmin, ymin], [xmax, ymax], (n_cibles, 2))
np.save("cibles.npy", cibles)  # ← ajoute cette ligne
n1 = int(0.33 * n_cibles)
n2 = int(0.33 * n_cibles)
n3 = n_cibles - n1 - n2
poids_cibles = np.array([3]*n1 + [2]*n2 + [1]*n3, dtype=float)
np.random.shuffle(poids_cibles)
np.save("poids_cibles.npy", poids_cibles)  # ← ajoute cette ligne

couleurs_cibles = np.array([
    'red' if p == 3 else 'orange' if p == 2 else 'green'
    for p in poids_cibles
])
# Sauvegarde (couleurs_cibles est un tableau de strings)
np.save("couleurs_cibles.npy", couleurs_cibles)

zone_global = box(xmin, ymin, xmax, ymax)

cibles_t = torch.tensor(cibles,       dtype=torch.float32).to(DEVICE)
poids_t  = torch.tensor(poids_cibles, dtype=torch.float32).to(DEVICE)

# ════════════════════════════════════════════════
#  CARTE DE PRIORITÉ
# ════════════════════════════════════════════════
def build_priority_map(cibles, poids_cibles, grid=GRID, sigma=3.0):
    scale = grid / 100.0
    pmap  = np.zeros((grid, grid), dtype=np.float32)
    for j, c in enumerate(cibles):
        gx = int(np.clip(c[0] * scale, 0, grid - 1))
        gy = int(np.clip(c[1] * scale, 0, grid - 1))
        pmap[gy, gx] += poids_cibles[j]
    pmap = gaussian_filter(pmap, sigma=sigma)
    if pmap.max() > 0:
        pmap /= pmap.max()
    return torch.tensor(pmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

PRIORITY_MAP = build_priority_map(cibles, poids_cibles)

# ════════════════════════════════════════════════
#  ATTENTION SPATIALE
# ════════════════════════════════════════════════
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv            = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.priority_weight = nn.Parameter(torch.tensor(2.0))
        self.last_attn       = np.zeros((GRID, GRID), dtype=np.float32)

    def forward(self, x, priority_map):
        if priority_map.shape[-2:] != x.shape[-2:]:
            priority_map = F.interpolate(priority_map, size=x.shape[-2:],
                                         mode='bilinear', align_corners=False)
        score          = self.conv(x)
        attn           = torch.sigmoid(score + self.priority_weight * priority_map)
        self.last_attn = attn.detach().cpu().numpy()[0, 0]
        return x * attn

# ════════════════════════════════════════════════
#  ACTOR : CentroidCNN
# ════════════════════════════════════════════════
class CentroidCNN(nn.Module):
    def __init__(self, n_drones):
        super().__init__()
        self.n_drones = n_drones
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.attention = SpatialAttention(in_channels=64)
        self.pool      = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, n_drones * 2),
            nn.Sigmoid()
        )

    def forward(self, x, priority_map):
        feat = self.encoder(x)
        feat = self.attention(feat, priority_map)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)

    def extend(self, new_n, init_pos_norm=None):
        if new_n == self.n_drones:
            return
        old_w = self.head[2].weight.data
        old_b = self.head[2].bias.data
        new_linear = nn.Linear(256, new_n * 2)
        nn.init.xavier_uniform_(new_linear.weight)
        nn.init.zeros_(new_linear.bias)
        new_linear.weight.data[:self.n_drones*2] = old_w
        new_linear.bias.data[:self.n_drones*2]   = old_b
        if init_pos_norm is not None:
            pos = np.clip(init_pos_norm, 0.01, 0.99).astype(np.float32)
            logit = np.log(pos / (1.0 - pos))
            new_linear.bias.data[self.n_drones*2]   = float(logit[0])
            new_linear.bias.data[self.n_drones*2+1] = float(logit[1])
        self.head[2]  = new_linear
        self.n_drones = new_n

# ════════════════════════════════════════════════
#  CRITIC : CoverageCNN
# ════════════════════════════════════════════════
class CoverageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 32),  nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.historique_pred = []
        self.historique_reel = []

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        return self.head(feat)

# ════════════════════════════════════════════════
#  SEUIL DE COÛT
# ════════════════════════════════════════════════
def calcul_seuil_initial(xmin, xmax, ymin, ymax, n_cibles, R, alpha, beta):
    surface_zone  = (xmax - xmin) * (ymax - ymin)
    densite       = n_cibles / surface_zone
    taux_attendu  = min(1.0, densite * np.pi * R**2)
    variance_ref  = densite * R**2
    return alpha * (1.0 - taux_attendu) + beta * variance_ref

COUT_SEUIL    = calcul_seuil_initial(xmin, xmax, ymin, ymax, n_cibles, R, alpha, beta)
densite_init  = n_cibles / ((xmax-xmin)*(ymax-ymin))
taux_att_init = min(1.0, densite_init * np.pi * R**2)
print(f"[Init] Seuil de coût prévisionnel (FIXE) : {COUT_SEUIL:.4f}")
print(f"       zone={xmax}×{ymax}  n_cibles={n_cibles}  R={R}"
      f"  densité={densite_init:.4f}/u²  taux_attendu={taux_att_init:.2%}")

# ════════════════════════════════════════════════
#  INITIALISATION
# ════════════════════════════════════════════════
actor      = CentroidCNN(n_drones).to(DEVICE)
actor_opt  = optim.Adam(actor.parameters(), lr=LR)

critic     = CoverageCNN().to(DEVICE)
critic_opt = optim.Adam(critic.parameters(), lr=LR_COV)

prev_cov_pred = None

W_COV      = 1.0
W_SPREAD   = 0.3
W_PRIORITY = 0.2
T_ASSIGN   = 3.0
SIGMA_SPRD = 20.0

_gy, _gx  = np.mgrid[0:GRID, 0:GRID]
_scale     = 100.0 / GRID
PIX_X      = torch.tensor(_gx * _scale + _scale/2, dtype=torch.float32).flatten().to(DEVICE)
PIX_Y      = torch.tensor(_gy * _scale + _scale/2, dtype=torch.float32).flatten().to(DEVICE)
PIXELS     = torch.stack([PIX_X, PIX_Y], dim=1)
PMAP_FLAT  = PRIORITY_MAP[0, 0].flatten().clamp(min=0)
PMAP_NP    = PRIORITY_MAP[0, 0].cpu().numpy()

historique_convergences = []
prev_cout = None

# ════════════════════════════════════════════════
#  CONSTRUCTION IMAGE
# ════════════════════════════════════════════════
def build_image(pos_drones, cibles, poids_cibles):
    img    = np.zeros((3, GRID, GRID), dtype=np.float32)
    scale  = GRID / 100.0
    r_grid = R * scale

    for j, c in enumerate(cibles):
        gx = int(np.clip(c[0] * scale, 0, GRID - 1))
        gy = int(np.clip(c[1] * scale, 0, GRID - 1))
        img[0, gy, gx] = poids_cibles[j] / 3.0

    for pos in pos_drones:
        gx = int(np.clip(pos[0] * scale, 0, GRID - 1))
        gy = int(np.clip(pos[1] * scale, 0, GRID - 1))
        img[1, gy, gx] = 1.0

    ys, xs = np.ogrid[:GRID, :GRID]
    for pos in pos_drones:
        cx, cy = pos[0] * scale, pos[1] * scale
        mask   = (xs - cx)**2 + (ys - cy)**2 <= r_grid**2
        img[2][mask] = 1.0

    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ════════════════════════════════════════════════
#  FONCTIONS MÉTIER
# ════════════════════════════════════════════════
def calcul_taux_couverture(idx):
    if len(idx) == 0:
        return 0.0
    return (np.sum(poids_cibles[idx]) / np.sum(poids_cibles)) * 100.0

def calcul_variance(cellules, drones):
    C = []
    for drone in drones.geoms:
        for cellule in cellules:
            if cellule.contains(drone):
                cercle = drone.buffer(R)
                ci = sum(1 for j, cible in enumerate(cibles)
                        # if cellule.contains(Point(cible)) and cercle.contains(Point(cible)) )
                          if cellule.contains(Point(cible)))
                C.append(ci)
                break
    C = np.array(C)
    if len(C) == 0:
        return 0.0, C
    return float(np.mean((C - C.mean())**2)), C

def deplacement(p_drone, p_dest):
    d    = p_dest - p_drone
    dist = np.linalg.norm(d)
    if dist < 1:
        return p_dest
    return p_drone + (d / dist) * 1.0

# ════════════════════════════════════════════════
#  BARYCENTRE PMAP NON COUVERTE
# ════════════════════════════════════════════════
def barycentre_pmap_non_couverte(pos_drones):
    scale  = GRID / 100.0
    r_grid = R * scale
    ys, xs = np.ogrid[:GRID, :GRID]
    couverts = np.zeros((GRID, GRID), dtype=bool)
    for pos in pos_drones:
        cx = pos[0] * scale
        cy = pos[1] * scale
        couverts |= ((xs - cx)**2 + (ys - cy)**2 <= r_grid**2)
    pmap_residuelle = PMAP_NP * (~couverts).astype(np.float32)
    total = pmap_residuelle.sum()
    if total < 1e-6:
        return np.array([0.5, 0.5])
    gx_coords = (_gx * (100.0/GRID) + (50.0/GRID)) / 100.0
    gy_coords = (_gy * (100.0/GRID) + (50.0/GRID)) / 100.0
    bary_x = (pmap_residuelle * gx_coords).sum() / total
    bary_y = (pmap_residuelle * gy_coords).sum() / total
    return np.clip(np.array([bary_x, bary_y]), 0.01, 0.99)

# ════════════════════════════════════════════════
#  LOSS ACTOR
# ════════════════════════════════════════════════
def actor_loss_cnn(pred, cov_pred_detached):
    D   = pred.shape[1] // 2
    pos = pred[0].reshape(D, 2) * 100.0

    diff     = cibles_t.unsqueeze(1) - pos.unsqueeze(0)
    dist2    = (diff**2).sum(-1)
    dist     = dist2.sqrt().clamp(min=1e-4)
    assign   = F.softmax(-dist2 / T_ASSIGN, dim=1)
    cov_soft = torch.sigmoid((R - dist) * 0.3)
    covered  = (assign * cov_soft).sum(dim=1)
    taux     = (covered * poids_t).sum() / poids_t.sum()
    L_cov    = 1.0 - taux

    L_spread = torch.tensor(0.0, device=DEVICE)
    if D > 1:
        ii, jj   = torch.triu_indices(D, D, offset=1)
        d2_ij    = ((pos[ii] - pos[jj])**2).sum(-1)
        L_spread = torch.exp(-d2_ij / (2 * SIGMA_SPRD**2)).mean()

    diff_pd  = PIXELS.unsqueeze(1) - pos.unsqueeze(0)
    d2_pd    = (diff_pd**2).sum(-1)
    assign_p = F.softmax(-d2_pd / (T_ASSIGN * 8), dim=1)
    w_px     = PMAP_FLAT.unsqueeze(1) * assign_p
    w_sum    = w_px.sum(0).clamp(min=1e-6)
    bx       = (w_px * PIX_X.unsqueeze(1)).sum(0) / w_sum
    by       = (w_px * PIX_Y.unsqueeze(1)).sum(0) / w_sum
    bary     = torch.stack([bx, by], dim=1)
    L_priority = F.mse_loss(pos, bary.detach()) / (100.0**2)

    loss = (W_COV * L_cov
            + W_SPREAD * L_spread
            + W_PRIORITY * L_priority
            - cov_pred_detached * 1e-2)

    return loss, L_cov.item(), L_spread.item(), L_priority.item()

# ════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 7))
ax        = fig.add_subplot(1, 3, 1)
ax_table  = fig.add_subplot(1, 3, 2)
ax_cov    = fig.add_subplot(1, 3, 3)
ax_table.axis('off')

variancee = 0.0

def update(frame):
    global positions_drones, centroides, n_drones
    global variancee, actor, actor_opt, prev_cout
    global historique_convergences, prev_cov_pred

    ax.clear()
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal')
    ax.plot(*zone_global.exterior.xy, 'b-', lw=1)

    drones           = MultiPoint(positions_drones)
    cellules         = voronoi_polygons(drones, extend_to=zone_global)
    cellules_coupees = [c.intersection(zone_global) for c in cellules.geoms]
    variancee, coutVar = calcul_variance(cellules_coupees, drones)


    # ── Taux réel géométrique ─────────────────────────────────────────────
    taux_drones = np.zeros(n_drones)
    for i, drone in enumerate(drones.geoms):
        cercle = drone.buffer(R)
        a_une_cible = any(cercle.covers(Point(cible)) for cible in cibles)
        
        if not a_une_cible:
           continue
        for cellule in cellules_coupees:
            if cellule.covers(drone):
               
                idx = [j for j, cible in enumerate(cibles)
                       if cellule.covers(Point(cible)) and cercle.covers(Point(cible))]
                taux_drones[i] = calcul_taux_couverture(np.array(idx, dtype=int))
                break

    taux_total = np.sum(taux_drones) / 100.0
    cout_total = alpha * (1 - taux_total) + beta * variancee

    img_tensor = build_image(positions_drones, cibles, poids_cibles)

    # ════ ÉTAPE 1 : Entraînement CRITIC ══════════════════════════════════
    cov_pred     = critic(img_tensor)
    cov_pred_val = cov_pred.item() * 100.0

    label_reel  = torch.tensor([[taux_total]], dtype=torch.float32).to(DEVICE)
    loss_critic = nn.MSELoss()(cov_pred, label_reel)

    critic_opt.zero_grad()
    loss_critic.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_opt.step()

    critic.historique_pred.append(cov_pred_val)
    critic.historique_reel.append(taux_total * 100.0)

    # ════ ÉTAPE 2 : Entraînement ACTOR ═══════════════════════════════════
    pred = actor(img_tensor, PRIORITY_MAP)

    loss_actor, lc, ls, lp = actor_loss_cnn(pred, cov_pred.detach())

    actor_opt.zero_grad()
    loss_actor.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    # ── Centroïdes prédits → déplacement ─────────────────────────────────
    pred_np        = pred.detach().cpu().numpy()[0]
    centroides_cnn = pred_np.reshape(n_drones, 2) * 100.0
    centroides_cnn = np.clip(centroides_cnn, [xmin, ymin], [xmax, ymax])

  

    for i in range(n_drones):
        positions_drones[i] = deplacement(positions_drones[i], centroides_cnn[i])

    # ════ CONVERGENCE ════════════════════════════════════════════════════
    dist_to_cnn = np.linalg.norm(positions_drones - centroides_cnn, axis=1)
    convergence  = np.all(dist_to_cnn < 2)

    if convergence:
        entree = {
            'frame'    : frame,
            'n_drones' : n_drones,
            'taux'     : taux_total * 100,
            'variance' : variancee,
            'cout'     : cout_total,
            'seuil'    : COUT_SEUIL,
            'cov_pred' : cov_pred_val,
        }
        if not historique_convergences or historique_convergences[-1]['frame'] != frame:
            historique_convergences.append(entree)
            print(f"[Convergence #{len(historique_convergences)}] "
                  f"Frame={frame} | Drones={n_drones} | "
                  f"Taux réel={taux_total*100:.2f}% | "
                  f"Critic={cov_pred_val:.2f}% | "
                  f"Variance={variancee:.3f} | "
                  f"Cout={cout_total:.4f} | "
                  f"Seuil={COUT_SEUIL:.4f}")

    if convergence and cout_total <= COUT_SEUIL:
        print(f"✅ STOP optimal | Cout={cout_total:.4f} | Seuil={COUT_SEUIL:.4f}")
        ani.event_source.stop()

    elif convergence and n_drones == MAX_DRONES:
        print(f"✅ MAX_DRONES atteint — {MAX_DRONES}")
        ani.event_source.stop()

    elif convergence and cout_total > COUT_SEUIL and n_drones < MAX_DRONES:
        # ════ AJOUT DRONE PAR CNN ════════════════════════════════════════
        init_norm = barycentre_pmap_non_couverte(positions_drones)
        init_pos  = init_norm * 100.0

        actor.extend(n_drones + 1, init_pos_norm=init_norm)
        actor_opt = optim.Adam(actor.parameters(), lr=LR)

        with torch.no_grad():
            pred_new = actor(img_tensor, PRIORITY_MAP)
        positions_predites = pred_new[0].cpu().numpy().reshape(n_drones + 1, 2) * 100.0
        positions_predites = np.clip(positions_predites, [xmin, ymin], [xmax, ymax])
        nouveau = positions_predites[-1]

        positions_drones = np.vstack([positions_drones, nouveau])
        centroides       = np.vstack([centroides, [0.0, 0.0]])
        n_drones         = len(positions_drones)
        prev_cov_pred    = None

        print(f"[+drone CNN] → {n_drones} drones | "
              f"Init PMAP=({init_pos[0]:.1f},{init_pos[1]:.1f}) | "
              f"Pos CNN=({nouveau[0]:.1f},{nouveau[1]:.1f}) | "
              f"Cout={cout_total:.4f}")

    # ════ AFFICHAGE ═══════════════════════════════════════════════════════
    for cellule in cellules_coupees:
        x, y = cellule.exterior.xy
        ax.fill(x, y, color='lightgray', edgecolor='black', linewidth=0.5)

    for drone in drones.geoms:
        cercle = drone.buffer(R)
        for cellule in cellules_coupees:
            if cellule.covers(drone):
                inter = cercle.intersection(cellule)
                if not inter.is_empty:
                    polys = [inter] if inter.geom_type == "Polygon" else inter.geoms
                    for poly in polys:
                        ax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
                break

    for pos in positions_drones:
        draw_drone(ax, pos[0], pos[1], size=3.5)
    ax.plot([], [], color='black', marker='o', markersize=6,
            linestyle='None', label='Drones')   # entrée légende
    ax.scatter(cibles[:, 0], cibles[:, 1],
               c=couleurs_cibles, s=30, marker='x', zorder=4)
    ax.scatter(centroides_cnn[:, 0], centroides_cnn[:, 1],
               color='purple', s=40, marker='*', zorder=6, label='CNN (centroïdes)')

    for i, pos in enumerate(positions_drones):
        ax.text(pos[0], pos[1] + 3, f"T:{taux_drones[i]:.1f}%", fontsize=8, color='blue')

    ax.text(0.02, 1.12, f"Couverture réelle : {taux_total*100:.2f}%",
            transform=ax.transAxes, color='red', fontsize=9)
    ax.text(0.02, 1.07, f"Critic (estimée)  : {cov_pred_val:.2f}%",
            transform=ax.transAxes, color='purple', fontsize=9)
    ax.text(0.02, 1.02,
            f"L_cov={lc:.4f} | L_spr={ls:.4f} | L_pri={lp:.4f} | "
            f"Coût={cout_total:.3f} | Seuil={COUT_SEUIL:.3f} | "
            f"Drones={n_drones} | Frame={frame}",
            transform=ax.transAxes, color='gray', fontsize=8)
   

    # ════ TABLEAU CONVERGENCES ════════════════════════════════════════════
    ax_table.clear()
    ax_table.axis('off')
    titre = "Historique des convergences"
    if n_drones >= MAX_DRONES:
        titre += f"  : MAX {MAX_DRONES} drones atteint"
    ax_table.set_title(titre, fontsize=10, fontweight='bold',
                       color='red' if n_drones >= MAX_DRONES else 'black')

    if historique_convergences:
        col_labels = ['#', 'Frame', 'Drones', 'Taux réel', 'Critic', 'Variance', 'Coût', 'Seuil']
        rows = []
        for i, e in enumerate(historique_convergences):
            rows.append([
                str(i + 1), str(e['frame']), str(e['n_drones']),
                f"{e['taux']:.1f}%", f"{e['cov_pred']:.1f}%",
                f"{e['variance']:.3f}",
                f"{e['cout']:.4f}", f"{e['seuil']:.4f}",
            ])
        cout_min    = min(e['cout'] for e in historique_convergences)
        cell_colors = [
            ['#d4f7d4'] * 8 if e['cout'] == cout_min else ['#ffffff'] * 8
            for e in historique_convergences
        ]
        table = ax_table.table(
            cellText=rows, colLabels=col_labels,
            cellColours=cell_colors, loc='center', cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.1, 1.5)
        ax_table.text(0.5, 0.02,
                      f"Ligne verte = meilleur coût  |  Seuil fixe = {COUT_SEUIL:.4f}",
                      ha='center', va='bottom', transform=ax_table.transAxes,
                      fontsize=7, color='green')
    else:
        ax_table.text(0.5, 0.5, "En attente de la\npremière convergence...",
                      ha='center', va='center', transform=ax_table.transAxes,
                      fontsize=10, color='gray')

    # ════ COURBE COUVERTURE ═══════════════════════════════════════════════
    ax_cov.clear()
    ax_cov.set_title("Critic (CoverageCNN) vs Couverture Réelle", fontsize=10)
    ax_cov.set_xlabel("Frame", fontsize=8)
    ax_cov.set_ylabel("Taux de couverture (%)", fontsize=8)
    ax_cov.set_ylim(0, 105)
    ax_cov.axhline(100, color='green', linestyle='--', linewidth=1, label='Objectif 100%')

    frames_x = list(range(len(critic.historique_reel)))
    if frames_x:
        ax_cov.plot(frames_x, critic.historique_reel,
                    color='red', linewidth=1.2, label='Réel', alpha=0.8)
        ax_cov.plot(frames_x, critic.historique_pred,
                    color='purple', linewidth=1.2, label='Critic',
                    alpha=0.8, linestyle='--')
        erreur = abs(critic.historique_pred[-1] - critic.historique_reel[-1])
        ax_cov.text(0.02, 0.08,
                    f"Critic : {critic.historique_pred[-1]:.1f}%\n"
                    f"Réel   : {critic.historique_reel[-1]:.1f}%\n"
                    f"Erreur : {erreur:.1f}%\n"
                    f"Loss critic : {loss_critic.item():.5f}\n"
                    f"Loss actor  : {loss_actor.item():.5f}\n"
                    f"Seuil fixe  : {COUT_SEUIL:.4f}",
                    transform=ax_cov.transAxes, fontsize=8, color='black',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

   
    ax_cov.grid(True, alpha=0.3)
    fig.tight_layout()


ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.tight_layout()
plt.show()