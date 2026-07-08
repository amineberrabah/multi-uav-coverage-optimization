

from shapely import voronoi_polygons, MultiPoint
from shapely.geometry import box, Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Paramètres globaux de la simulation ───────────────────────────────────────
np.random.seed(44)
R = 20          # Rayon de détection de chaque drone
n_cibles = 100   # Nombre initial de cibles

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

# Position initiale aléatoire (remplacée par des valeurs fixes issues d'une convergence)
positions_drones = np.random.uniform([xmin, ymin], [xmax, ymax], (2, 2))

# ── Positions fixes des 4 drones ──────────────────────────────────────────────
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
# Cible actuellement suivie par chaque drone (None = pas de cible assignée)
cible_courante = [None] * len(positions_drones)
# Nombre d'itérations de cooldown restantes pour chaque cible
cooldown = np.zeros(n_cibles)

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


# ── Pondération des cibles (priorité 1, 2 ou 3) ───────────────────────────────
n1 = int(0.33 * n_cibles)
n2 = int(0.33 * n_cibles)
n3 = n_cibles - n1 - n2

poids_cibles = np.array([3]*n1 + [2]*n2 + [1]*n3)
np.random.shuffle(poids_cibles)
# nombre_visite[i] = nombre de passages restants requis sur la cible i
nombre_visite = np.ones(n_cibles) * poids_cibles

# Couleurs d'affichage liées au poids (rouge = haute priorité)
cc = []
for p in poids_cibles:
    if p == 3:
        cc.append('red')
    elif p == 2:
        cc.append('orange')
    else:
        cc.append('green')
cc = np.array(cc)
couleurs_cibles = cc   # tableau modifié dynamiquement (gris = épuisé, bleu = en cours)

fig, ax = plt.subplots(figsize=(6,6))

# ── Paramètres des paquets de données et de la dynamique des cibles ───────────
paquets = []
PAQUET_VITESSE = 2.0   # unités par frame
PAQUET_TTL     = 80    # durée de vie maximale d'un paquet (frames)
R_base = 1             # rayon de base du potentiel élastique
alpha = 1              # coefficient d'élargissement temporel du potentiel
eta = 1                # pas de déplacement selon le gradient
temps = 0
MAX_NOUVELLES_CIBLES = 20    # plafond de cibles ajoutées dynamiquement
INTERVAL_AJOUT = 20          # intervalle en frames entre deux ajouts
nb_cibles_ajoutees = 0
# Mémorise la frame de dernière visite pour chaque cible (calcul du délai)
derniere_visite = np.zeros(n_cibles)

def ajouter_nouvelle_cible():
    # Ajoute une cible à position et priorité aléatoires ; met à jour tous les tableaux globaux
    global cibles, poids_cibles, nombre_visite, couleurs_cibles, cc, cooldown, derniere_visite, n_cibles, nb_cibles_ajoutees
    nouvelle_cible = np.random.uniform([xmin, ymin], [xmax, ymax])
    nouveau_poids = np.random.choice([1, 2, 3])
    cibles = np.vstack([cibles, nouvelle_cible])
    poids_cibles = np.append(poids_cibles, nouveau_poids)
    nombre_visite = np.append(nombre_visite, nouveau_poids)
    cooldown = np.append(cooldown, 0)
    derniere_visite = np.append(derniere_visite, 0)

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

def largeur_elastique(i):
    # Plus une cible est longtemps non visitée, plus son champ d'attraction s'élargit
    delai = temps - derniere_visite[i]
    return R_base + alpha * np.sqrt(delai + 1)

def potentiel_cible(position, i):
    # Potentiel gaussien centré sur la cible i ; nul si cooldown actif ou cible épuisée
    if cooldown[i] > 0:
        return 0
    if nombre_visite[i] <= 0:
        return 0
    sigma = largeur_elastique(i)
    dist2 = np.sum((position - cibles[i])**2)
    return poids_cibles[i] * np.exp(-dist2 / (2 * sigma**2))

def potentiel_total(position, indices):
    # Somme des potentiels de toutes les cibles actives dans la liste d'indices
    total = 0
    for i in indices:
        total += potentiel_cible(position, i)
    return total

def gradient_potentiel(position, indices, h=1.0):
    # Gradient numérique du potentiel total (différences finies centrées)
    grad = np.zeros(2)
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        grad[k] = (
            potentiel_total(position + e, indices) -
            potentiel_total(position - e, indices)
        ) / (2 * h)
    return grad

def prochaine_position(position, indices, d):
    # Calcule la prochaine position du drone d selon le gradient ou une cible assignée
    if all(nombre_visite[i] <= 0 or cooldown[i] > 0 for i in indices):
        return position, 0 # 🚀 le drone reste sur place
    
    # Si une cible est déjà assignée, on continue vers elle directement
    if cible_courante[d] is not None:
        cible = cibles[cible_courante[d]]
        return deplacement(position, cible)

    grad = gradient_potentiel(position, indices)

    norm = np.linalg.norm(grad)

    if norm == 0:
        # Gradient nul → fallback : on choisit la cible active la plus proche
        distances = []
        for i in indices:
            if  nombre_visite[i] > 0 and cooldown[i] == 0:
                dist = np.linalg.norm(position - cibles[i])
                distances.append(dist)
            else:
                distances.append(1e6)

        best = indices[np.argmin(distances)]
        cible_courante[d] = best

        return deplacement(position, cibles[best])
    
    # Déplacement d'un pas dans la direction du gradient montant
    direction = grad / norm
    new_pos = position + eta * direction
    dist = np.linalg.norm(new_pos - position)
    
    return new_pos, dist

def trouver_cible_proche(position, indices, seuil=2):
    # Retourne l'index de la première cible à moins de `seuil` unités, ou None
    for i in indices:
        if np.linalg.norm(position - cibles[i]) < seuil:
            return i
    return None

def visiter(d, j):
    # Décrémente le compteur de visite de la cible j et déclenche son cooldown
    if cible_courante[d] == j:
        i = cible_courante[d]
    else:
        i = j
    if cooldown[i] == 0 and nombre_visite[i] > 0:
        derniere_visite[i] = temps
        nombre_visite[i] -= 1
        cooldown[i] = 10
    # Mise à jour de la couleur : gris = épuisée, bleu = en cours de visites
    if nombre_visite[i] <= 0:
        couleurs_cibles[i] = 'grey'  
    else:
        couleurs_cibles[i] = 'blue'
    cible_courante[d] = None

# ── Paramètres de la station de base et du modèle d'énergie ──────────────────
# =========================
station_base = np.array([50, 50])
cooldown_drone = np.zeros(len(positions_drones))     # cooldown de recharge par drone
en_recharge = np.zeros(len(positions_drones), dtype=bool)
H = 20                                               # altitude de vol (distance 3D)
energie_drones = np.full(len(positions_drones), 200000.0)
energie_consommee = np.zeros(len(positions_drones))
energie_comm_consommee = np.zeros(len(positions_drones))   # énergie communication cumulée par drone

# Coefficients du modèle de puissance aérodynamique
a = 1
b = 0.3
c = 0.5
Psense = 5    # puissance capteur

def calcul_energie_comm(p_drone, station=station_base):
    # Énergie de communication ∝ carré de la distance 3D à la station
    dx = p_drone[0] - station[0]
    dy = p_drone[1] - station[1]
    distance_3d = np.sqrt(dx**2 + dy**2 + H**2)
    return distance_3d**2


def calcul_energie(distance, v=10):
    # Énergie de déplacement = puissance × temps de vol
    delta_t = distance / v
    puissance = a + b*(v**2) + c*(v**3) + Psense
    return puissance * delta_t 

def deplacement(p_drone, p_distination):
    # Avance le drone d'un pas unitaire vers la destination ; retourne (nouvelle position, distance)
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
    
def update(frame):
    global positions_drones, temps, nombre_visite, couleurs_cibles

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
            # Collecte des cibles présentes dans la cellule Voronoï du drone
            indices = []
            for j in range(n_cibles):
                if cellules_coupees[i].contains(Point(cibles[j])):
                    indices.append(j)
                
            # Aucune cible dans la cellule → le drone attend
            if len(indices) == 0:
                continue

            # Calcul de la prochaine position via champ de potentiel
            new_pos, dist_parcourue = prochaine_position(positions_drones[i], indices, i)
            positions_drones[i] = new_pos
            
            # Consommation d'énergie : hovering si immobile, vol sinon
            if dist_parcourue == 0:
                energie_consommee[i] += Psense * a
                energie_drones[i] -= Psense * a
            else:
                E = calcul_energie(dist_parcourue)
                energie_drones[i] -= E
                energie_consommee[i] += E
            
            # Détection des cibles dans le rayon R → déclenchement de la visite
            drone = Point(positions_drones[i])
            cercle = drone.buffer(R)
            for j in indices: 
                cible = cibles[j] 
                if cellules_coupees[i].covers(Point(cible)) and cercle.covers(Point(cible)):
                    visiter(i, j)
            
            # Toutes les cibles épuisées et hors cooldown → envoi paquet + réinitialisation du cycle
            if all(nombre_visite[j] <= 0 and cooldown[j] == 0 for j in indices):
                paquets.append({
                    'pos':      positions_drones[i].copy(),
                    'origine':  positions_drones[i].copy(),
                    'drone_id': i,
                    'ttl':      PAQUET_TTL
                })
                    
                # Coût énergétique de la transmission radio
                E_comm = calcul_energie_comm(new_pos)
                energie_drones[i] -= E_comm
                energie_comm_consommee[i] += E_comm 
                energie_consommee[i] += E_comm
                
                # Réinitialisation complète du cycle pour toutes les cibles de la cellule
                for j in indices:
                    nombre_visite[j] = poids_cibles[j]
                    couleurs_cibles[j] = cc[j]
                    cooldown[j] = 0
                    derniere_visite[j] = 0  
                        
            # Mise à jour des couleurs selon l'état de cooldown de chaque cible
            for j in indices:
                if cooldown[j] == 0 and not nombre_visite[j] <= 0:
                    if poids_cibles[j] == 1:
                        couleurs_cibles[j] = 'green'
                    if poids_cibles[j] == 2:
                        couleurs_cibles[j] = 'orange'
                    if poids_cibles[j] == 3:
                        couleurs_cibles[j] = 'red'
                if cooldown[j] > 0:
                    cooldown[j] -= 1   # décompte du cooldown à chaque frame
        else:
            # ── Énergie critique : retour à la station de base ────────────────

            if not np.allclose(positions_drones[i], station_base, atol=1):
                # Le drone n'est pas encore arrivé → il avance vers la station
                new_pos, dist_parcourue = deplacement(positions_drones[i], station_base)
                positions_drones[i] = new_pos

                E = calcul_energie(dist_parcourue)
                energie_drones[i] -= E
                energie_consommee[i] += E

            else:
                # Arrivée à la station → déclenchement de la recharge
                if not en_recharge[i]:
                    cooldown_drone[i] = 11
                    en_recharge[i] = True

                # Décompte du cooldown de recharge ; énergie restaurée à expiration
                if cooldown_drone[i] > 0:
                    cooldown_drone[i] -= 1
                else:
                    energie_drones[i] = 200000
                    en_recharge[i] = False            
        
    # ── Affichage des contours des cellules de Voronoï ────────────────────────
    for cellule in cellules_coupees:
        x, y = cellule.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.5)
        
    # Marqueur de la station de base
    ax.scatter(station_base[0], station_base[1], color='red', s=100, marker='^')
    
    # Lignes pointillées reliant chaque drone actif à la station
    for i, pos in enumerate(positions_drones):
        if energie_drones[i] > 0:
            ax.plot([pos[0], station_base[0]], [pos[1], station_base[1]],
                    color='red', linewidth=0.5, alpha=0.8, linestyle='--')
            
            
    # # =========================
    # # 🔵 HEATMAP PAR CELLULE
    # # =========================

    # resolution = 30

    # for cellule in cellules_coupees:

    #     # récupérer les cibles de cette cellule
    #     indices = []
    #     for i in range(n_cibles):
    #         if cellule.contains(Point(cibles[i])):
    #             indices.append(i)

    #     if len(indices) == 0:
    #         continue

    #     # grille
    #     x = np.linspace(xmin, xmax, resolution)
    #     y = np.linspace(ymin, ymax, resolution)
    #     X, Y = np.meshgrid(x, y)

    #     Z = np.zeros_like(X)

    #     for i in range(resolution):
    #         for j in range(resolution):
    #             pos = np.array([X[i, j], Y[i, j]])

    #             # ⚠️ potentiel seulement des cibles de la cellule
    #             if cellule.contains(Point(pos)):
    #                 Z[i, j] = potentiel_total(pos, indices)
    #             else:
    #                 Z[i, j] = np.nan  # masque hors cellule

    #     # # normalisation locale
    #     # if np.nanmax(Z) > 0:
    #     #     Z = Z / np.nanmax(Z)

    #     # affichage
    #     ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)

    # ── Animation des paquets de données en transit ───────────────────────────
    paquets_actifs = []
    for pkt in paquets:
        direction = station_base - pkt['pos']
        dist = np.linalg.norm(direction)
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

    paquets.clear()
    paquets.extend(paquets_actifs)
    
    # ── Affichage des drones et des cibles ────────────────────────────────────
    for pos in positions_drones:
        draw_drone(ax, pos[0], pos[1], size=3.5)
    
    ax.scatter(cibles[:,0],
               cibles[:,1],
               c=couleurs_cibles, s=30, marker='x')
    
    # Tracé du contour du cercle de détection de chaque drone (intersecté à sa cellule)
    for i in range(len(positions_drones)):
        drone = Point(positions_drones[i])
        cercle = drone.buffer(R)      
        partie = cercle.intersection(cellules_coupees[i])

        if not partie.is_empty:
            if partie.geom_type == "Polygon":
                x, y = partie.exterior.xy
                ax.plot(x, y, color='black', linewidth=0.5)

            elif partie.geom_type == "MultiPolygon":
                for poly in partie.geoms:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color='black', linewidth=0.5)
    
    # Affichage du cooldown restant au-dessus de chaque cible
    for i in range(n_cibles):
    #     # calcul du potentiel vu par le drone (ex: drone 0)
    #     pot = potentiel_cible(positions_drones[0], i)

    #     ax.text(
    #         cibles[i][0],
    #         cibles[i][1] + 2,   # petit décalage vertical
    #         f"{pot:.2f}",
    #         fontsize=8,
    #         color='black',
    #         ha='center'
    #     )
        ax.text(
            cibles[i][0],
            cibles[i][1] + 2,
            f"{int(cooldown[i])}",
            fontsize=6,
            color='black'
            )
        
    # Affichage de l'énergie consommée cumulée au-dessus de chaque drone
    for i in range(len(positions_drones)):
        ax.text(
            positions_drones[i][0],
            positions_drones[i][1] + 3,
            f"E:{energie_consommee[i]:.1f}",
            fontsize=8,
            color='black'
        )
        
    
# ── Lancement de l'animation (50 ms par frame, sans mise en cache) ────────────
ani = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
plt.show()