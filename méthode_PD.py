# ============================================================
#  Simulation de patrouille multi-drones avec gestion d'énergie
#  Les drones visitent des cibles dans leurs cellules de Voronoï,
#  envoient des paquets de données à la station de base,
#  et retournent se recharger quand leur énergie est faible.
# ============================================================

from shapely import voronoi_polygons, MultiPoint
from shapely.geometry import box, Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Initialisation de la zone et des paramètres de base ───────────────────────
np.random.seed(44)
n_cibles = 100

# ════════════════════════════════════════════════
#  FONCTION DESSIN DRONE (forme quadrirotor noir)
# ════════════════════════════════════════════════
def draw_drone(ax, x, y, size=3.5):
    """Dessine un drone quadrirotor noir centré en (x, y)."""
    arm   = size * 0.8
    r_mot = size * 0.35
    r_bod = size * 0.28

    for angle in [45, 135]:
        dx = arm * np.cos(np.radians(angle))
        dy = arm * np.sin(np.radians(angle))
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color='black', lw=1.5, zorder=6, solid_capstyle='round')

    for angle in [45, 135, 225, 315]:
        mx = x + arm * np.cos(np.radians(angle))
        my = y + arm * np.sin(np.radians(angle))
        ax.add_patch(plt.Circle((mx, my), r_mot, color='black', fill=True, zorder=7))
        ax.add_patch(plt.Circle((mx, my), r_mot * 0.45, color='white', fill=True, zorder=8))

    ax.add_patch(plt.Circle((x, y), r_bod, color='black', fill=True, zorder=9))

xmin, xmax = 0, 100
ymin, ymax = 0, 100
zone_global = box(xmin, ymin, xmax, ymax)

# Position initiale aléatoire (remplacée juste après par des valeurs fixes)
positions_drones = np.random.uniform([xmin, ymin], [xmax, ymax], (2, 2))

# ── Positions fixes des 4 drones (issues d'une convergence préalable) ─────────
positions_drones = np.array([
    [15.43294525, 22.44923782],
    [89.28302765, 78.53681183],
    [84.64440918, 10.39385033],
    [16.75672722, 87.02253723],
    [60.19989395, 10.69089985],
    [13.83406353, 54.04789352],
    [59.85033417, 86.16716766],
    [41.0093956,  64.22096252],
    [85.3409729,  42.36255646],
    [45.5923747,  41.11096793]])
# Génération aléatoire des cibles dans la zone
cibles = np.random.uniform([xmin, ymin], [xmax, ymax], (n_cibles, 2))
# 1 = cible active (non visitée), 0 = cible visitée
visite = np.ones((len(cibles)))

# ── Construction de la tessellation de Voronoï et association drone ↔ cellule ─
drones = MultiPoint(positions_drones)
cellules = voronoi_polygons(drones, extend_to=zone_global)
cellules_coupees_no = [cellule.intersection(zone_global) for cellule in cellules.geoms]
cellules_coupees = []
# Réordonne les cellules pour que cellules_coupees[i] corresponde au drone i
for i in range(len(positions_drones)):
    for cellule in cellules_coupees_no:
        if cellule.contains(Point(positions_drones[i])):
            cellules_coupees.append(cellule)

# ── Répartition des cibles en 3 niveaux de priorité (poids 3, 2, 1) ───────────
n1 = int(0.33 * n_cibles)
n2 = int(0.33 * n_cibles)
n3 = n_cibles - n1 - n2

poids_cibles = np.array([3]*n1 + [2]*n2 + [1]*n3)
np.random.shuffle(poids_cibles)

# Couleurs d'affichage associées aux niveaux de priorité
cc = []
for p in poids_cibles:
    if p == 3:
        cc.append('red')
    elif p == 2:
        cc.append('orange')
    else:
        cc.append('green')
cc = np.array(cc)
couleurs_cibles = cc.copy()   # copie modifiable (grisage à la visite)

fig, ax = plt.subplots(figsize=(6,6))

# ── Paramètres de dynamique des cibles (ajout périodique) ─────────────────────
temps = 0
MAX_NOUVELLES_CIBLES = 20    # nombre maximal de cibles dynamiques ajoutées
INTERVAL_AJOUT = 20          # intervalle en frames entre deux ajouts
nb_cibles_ajoutees = 0

# Station de base : point de recharge et destination des paquets de données
station_base = np.array([50, 50])  # centre

# Compteurs de recharge par drone
cooldown = np.zeros(len(positions_drones))
en_recharge = np.zeros(len(positions_drones), dtype=bool)

# File des paquets de données en transit vers la station de base
paquets = []
PAQUET_VITESSE = 2.0   # unités par frame
PAQUET_TTL     = 80    # durée de vie maximale d'un paquet (en frames)

def ajouter_nouvelle_cible():
    # Ajoute une cible à position et priorité aléatoires dans la zone
    global cibles, poids_cibles, couleurs_cibles, cc, n_cibles, nb_cibles_ajoutees, visite
    nouvelle_cible = np.random.uniform([xmin, ymin], [xmax, ymax])
    nouveau_poids = np.random.choice([1, 2, 3])
    cibles = np.vstack([cibles, nouvelle_cible])
    poids_cibles = np.append(poids_cibles, nouveau_poids)
    visite = np.append(visite, 1)
    
    if nouveau_poids == 1:
        couleur = 'green'
    elif nouveau_poids == 2:
        couleur = 'orange'
    else:
        couleur = 'red'

    couleurs_cibles = np.append(couleurs_cibles, couleur)
    cc = np.append(cc, couleur)
    n_cibles += 1
    nb_cibles_ajoutees += 1
    
# ── Modèle d'énergie des drones ────────────────────────────────────────────────
H = 20                              # altitude de vol (pour le calcul de distance 3D)
energie_drones = np.full(len(positions_drones), 200000.0)   # énergie initiale (unités)
energie_consommee = np.zeros(len(positions_drones))
energie_comm_consommee = np.zeros(len(positions_drones))   # énergie communication cumulée par drone

# Coefficients du modèle de puissance aérodynamique
a = 1
b = 0.3
c = 0.5
Psense = 5   # puissance consommée par le capteur

def calcul_energie_comm(p_drone, station=station_base):
    # Énergie de communication proportionnelle au carré de la distance 3D à la station
    dx = p_drone[0] - station[0]
    dy = p_drone[1] - station[1]
    distance_3d = np.sqrt(dx**2 + dy**2 + H**2)
    return distance_3d**2


def calcul_energie(distance, v=10):
    # Énergie de déplacement = puissance × temps (modèle polynomial en vitesse)
    delta_t = distance / v
    puissance = a + b*(v**2) + c*(v**3) + Psense
    return puissance * delta_t 

def deplacement(p_drone, p_distination):
    # Déplace le drone d'un pas unitaire vers la destination ; retourne la nouvelle position et la distance parcourue
    dir = p_distination - p_drone
    distance = np.linalg.norm(dir)

    if distance < 1:
        nouveau_p_drone = p_distination
        dist_parcourue = distance
    else:
        dir = dir / distance
        nouveau_p_drone = p_drone + dir * 1
        dist_parcourue = np.linalg.norm(p_drone - nouveau_p_drone)

    return nouveau_p_drone, dist_parcourue
    
def calcul_score(position, indices):
    # Calcule le score poids/distance pour chaque cible candidate et retourne l'index du meilleur
    scores = []
    for i in indices:
        Dij = np.linalg.norm(cibles[i] - position)
        score = poids_cibles[i] / (Dij + 1e-6)
        scores.append(score)
    
    return np.argmax(scores)
        
def update(frame):
    global temps

    temps += 1
    
    # Ajout périodique d'une nouvelle cible dynamique
    if (temps % INTERVAL_AJOUT == 0 and nb_cibles_ajoutees < MAX_NOUVELLES_CIBLES):
        ajouter_nouvelle_cible()
        
    ax.clear()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    zx, zy = zone_global.exterior.xy
    ax.plot(zx, zy)
    
    # ── Boucle principale : comportement de chaque drone ──────────────────────
    for i in range(len(positions_drones)):
        
        if energie_drones[i] > 10000:
            # Collecte des indices de cibles dans la cellule du drone
            indices_cellule = []
            indices_actifs = []
            for j in range(n_cibles):
                if cellules_coupees[i].contains(Point(cibles[j])):
                    indices_cellule.append(j)
                    if visite[j] == 1:
                        indices_actifs.append(j)
                
            # Aucune cible active dans la cellule → le drone attend
            if len(indices_actifs) == 0:
                continue
                
            # Sélection de la meilleure cible selon le score poids/distance
            best = calcul_score(positions_drones[i], indices_actifs)
            
            new_pos, dist_parcourue = deplacement(positions_drones[i], cibles[indices_actifs[best]])
            positions_drones[i] = new_pos
            
            # Consommation d'énergie : hovering si sur place, déplacement sinon
            if dist_parcourue == 0:
                energie_consommee[i] += Psense * a
                energie_drones[i] -= Psense * a
            else:
                E = calcul_energie(dist_parcourue)
                energie_drones[i] -= E
                energie_consommee[i] += E 
                
            # Marquage de la cible comme visitée (grisée sur l'affichage)
            if np.allclose(positions_drones[i], cibles[indices_actifs[best]]) and visite[indices_actifs[best]] == 1:
                visite[indices_actifs[best]] = 0
                couleurs_cibles[indices_actifs[best]] = 'grey'
                
            # Toutes les cibles de la cellule visitées → envoi d'un paquet + réinitialisation
            if all(visite[j] == 0 for j in indices_cellule):
                paquets.append({
                    'pos':      positions_drones[i].copy(),
                    'origine':  positions_drones[i].copy(),
                    'drone_id': i,
                    'ttl':      PAQUET_TTL
                })
                    
                # Coût énergétique de la transmission vers la station de base
                E_comm = calcul_energie_comm(new_pos)
                energie_drones[i] -= E_comm
                energie_comm_consommee[i] += E_comm 
                energie_consommee[i] += E_comm
                
                # Réactivation de toutes les cibles de la cellule pour un nouveau cycle
                for j in indices_cellule:
                    visite[j] = 1
                    couleurs_cibles[j] = cc[j]
                
        else:
            # ── Énergie critique : retour à la station de base pour recharge ──

            if not np.allclose(positions_drones[i], station_base, atol=1):
                # Le drone n'est pas encore à la station → il s'en approche
                new_pos, dist_parcourue = deplacement(positions_drones[i], station_base)
                positions_drones[i] = new_pos

                E = calcul_energie(dist_parcourue)
                energie_drones[i] -= E
                energie_consommee[i] += E

            else:
                # Le drone est arrivé à la station → déclenchement de la recharge
                if not en_recharge[i]:
                    cooldown[i] = 11
                    en_recharge[i] = True

                # Décompte du cooldown ; recharge complète à expiration
                if cooldown[i] > 0:
                    cooldown[i] -= 1
                else:
                    energie_drones[i] = 200000
                    en_recharge[i] = False
                        
    # ── Affichage des cellules de Voronoï ─────────────────────────────────────
    for cellule in cellules_coupees:
        x, y = cellule.exterior.xy
        ax.fill(x, y, color='lightgray', edgecolor='black', linewidth=0.5)
        
    # Marqueur de la station de base (triangle rouge)
    ax.scatter(station_base[0], station_base[1], color='red', s=100, marker='^')
    
    # Lignes pointillées reliant chaque drone actif à la station de base
    for i, pos in enumerate(positions_drones):
        if energie_drones[i] > 0:
            ax.plot([pos[0], station_base[0]], [pos[1], station_base[1]],
                    color='red', linewidth=0.5, alpha=0.8, linestyle='--')

    # for i in range(len(positions_drones)):
    #     drone = Point(positions_drones[i])
    #     cercle = drone.buffer(R)      
    #     partie = cercle.intersection(cellules_coupees[i])

    #     if not partie.is_empty:
    #         if partie.geom_type == "Polygon":
    #             x, y = partie.exterior.xy
    #             ax.plot(x, y, color='black', linewidth=0.5)

    #         elif partie.geom_type == "MultiPolygon":
    #             for poly in partie.geoms:
    #                 x, y = poly.exterior.xy
    #                 ax.plot(x, y, color='black', linewidth=0.5)
    
    # ── Animation des paquets de données en transit vers la station ───────────
    paquets_actifs = []
    for pkt in paquets:
        direction = station_base - pkt['pos']
        dist = np.linalg.norm(direction)
        # Paquet arrivé ou expiré → suppression (non ajouté à paquets_actifs)
        if dist < PAQUET_VITESSE or pkt['ttl'] <= 0:
            # paquet arrivé → on le supprime (pas d'ajout à paquets_actifs)
            continue
        direction = direction / dist
        pkt['pos'] = pkt['pos'] + direction * PAQUET_VITESSE
        pkt['ttl'] -= 1
        paquets_actifs.append(pkt)

        # dessin : petit carré violet animé
        px, py = pkt['pos']
        ax.scatter(px, py, color='purple', s=25, marker='s', zorder=7)

        # traînée : ligne de l'origine au paquet (atténuée)
        ax.plot([pkt['origine'][0], px], [pkt['origine'][1], py],
                color='purple', linewidth=0.8, alpha=0.3, linestyle='-')

        # label "PKT" discret
        ax.text(px + 1, py + 1, "PKT", fontsize=6, color='purple', zorder=8)

    # Remplacement de la liste de paquets par les paquets encore actifs
    paquets.clear()
    paquets.extend(paquets_actifs)
    
    # ── Affichage des drones, des cibles et des étiquettes d'énergie ──────────
    for pos in positions_drones:
        draw_drone(ax, pos[0], pos[1], size=3.5)
    
    ax.scatter(cibles[:,0],
               cibles[:,1],
               c=couleurs_cibles, s=30, marker='x')
    
    # Affichage du niveau d'énergie restant au-dessus de chaque drone
    for i in range(len(positions_drones)):
        ax.text(
            positions_drones[i][0],
            positions_drones[i][1] + 3,
            f"E:{energie_drones[i]:.1f}",
            fontsize=8,
            color='black'
        )
        # ax.text(
        #     positions_drones[i][0],
        #     positions_drones[i][1] + 2,
        #     f"{int(cooldown[i])}",
        #     fontsize=6,
        #     color='black'
        #     )
    
    
# ── Lancement de l'animation (50 ms par frame, sans mise en cache) ────────────
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()