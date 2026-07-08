from shapely import voronoi_polygons, MultiPoint
from shapely.geometry import box, Point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(44)
n_drones = 10
n_cibles = 100

xmin, xmax = 0, 100
ymin, ymax = 0, 100
temps = 0
MAX_NOUVELLES_CIBLES = 1
INTERVAL_AJOUT = 20
nb_cibles_ajoutees = 0

# ════════════════════════════════════════════════
#  FONCTION DESSIN DRONE (forme quadrirotor)
# ════════════════════════════════════════════════
def draw_drone(ax, x, y, size=3.5, body_color='black', arm_color='black'):
    """Dessine un drone quadrirotor centré en (x, y)."""
    arm   = size * 0.8
    r_mot = size * 0.35
    r_bod = size * 0.28

    for angle in [45, 135]:
        dx = arm * np.cos(np.radians(angle))
        dy = arm * np.sin(np.radians(angle))
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color=arm_color, lw=1.5, zorder=6, solid_capstyle='round')

    for angle in [45, 135, 225, 315]:
        mx = x + arm * np.cos(np.radians(angle))
        my = y + arm * np.sin(np.radians(angle))
        ax.add_patch(plt.Circle((mx, my), r_mot, color=body_color, fill=True, zorder=7))
        ax.add_patch(plt.Circle((mx, my), r_mot * 0.45, color='white', fill=True, zorder=8))

    ax.add_patch(plt.Circle((x, y), r_bod, color=body_color, fill=True, zorder=9))

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

cibles = np.load("cibles.npy")
centroides = np.zeros((n_drones, 2))

n1 = int(0.33 * n_cibles)
n2 = int(0.33 * n_cibles)
n3 = n_cibles - n1 - n2

poids_cibles = np.array([3]*n1 + [2]*n2 + [1]*n3, dtype=float)
np.random.shuffle(poids_cibles)

poids_cibles    = np.load("poids_cibles.npy")
couleurs_cibles = np.load("couleurs_cibles.npy", allow_pickle=True)

visites_requises   = poids_cibles.astype(int)
visites_effectuees = np.zeros(n_cibles, dtype=int)

zone_global      = box(xmin, ymin, xmax, ymax)
drones           = MultiPoint(positions_drones)
cellules         = voronoi_polygons(drones, extend_to=zone_global)
cellules_coupees = [cel.intersection(zone_global) for cel in cellules.geoms]

fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(12, 6),
                                    gridspec_kw={'width_ratios': [2, 1]})

energie_drones         = np.full(n_drones, 200000.0)
energie_consommee      = np.zeros(n_drones)
energie_comm_consommee = np.zeros(n_drones)

vitesse = 10
a       = 1
b       = 0.3
c       = 0.5
Psense  = 5

station       = np.array([50.0, 50.0])
h             = 20.0
p_comm        = 0.5
comm_interval = 10

SEUIL_ENERGIE_CRITIQUE = 1000
en_recharge            = np.zeros(n_drones, dtype=bool)
cooldown               = np.zeros(n_drones, dtype=int)
drone_en_retour        = np.zeros(n_drones, dtype=bool)

derniere_visite_cibles     = np.full(n_cibles, -np.inf)
cible_courante             = np.full(n_drones, -1, dtype=int)
drone_mission_terminee     = np.zeros(n_drones, dtype=bool)
frame_visite_drone         = np.full(n_drones, -1, dtype=int)
drone_en_attente_centroide = np.zeros(n_drones, dtype=bool)

frame_fin_mission = np.full(n_drones, -1, dtype=int)
duree_mission     = np.full(n_drones, -1, dtype=int)
frame_debut       = 0

historique_temps   = [[] for _ in range(n_drones)]
historique_energie = [[] for _ in range(n_drones)]
energie_tour       = np.zeros(n_drones)
tour_en_cours      = np.zeros(n_drones)
drone_au_centroide = np.ones(n_drones, dtype=bool)


def cibles_incomplete(indices):
    return [j for j in indices if visites_effectuees[j] < visites_requises[j]]

def toutes_cibles_completes_cellule(indices):
    return len(cibles_incomplete(indices)) == 0

def distance_3D(pos_drone, pos_station, hauteur):
    dx = pos_drone[0] - pos_station[0]
    dy = pos_drone[1] - pos_station[1]
    return np.sqrt(dx**2 + dy**2 + hauteur**2)

def calcul_energie_comm(pos_drone, pos_station=station, hauteur=h, p=p_comm):
    D = distance_3D(pos_drone, pos_station, hauteur)
    return p * (D ** 2)

def calcul_energie(distance, v=vitesse, _a=a, _b=b, _c=c, _Ps=Psense):
    if distance < 1e-9:
        return 0.0
    delta_t   = distance / v
    puissance = _a + _b*(v**2) + _c*(v**3) + _Ps
    return puissance * delta_t

def deplacement(p_drone, p_dest):
    direction = p_dest - p_drone
    distance  = np.linalg.norm(direction)
    if distance < 1:
        return p_dest.copy(), distance
    direction /= distance
    nouveau    = p_drone + direction * (vitesse / 10)
    return nouveau, np.linalg.norm(nouveau - p_drone)

def deplacer_si_energie(i, destination):
    global positions_drones, energie_drones, energie_consommee, \
           energie_tour, tour_en_cours
    nouveau_p, dist = deplacement(positions_drones[i], destination)
    E = calcul_energie(dist)
    energie_drones[i]    -= E
    energie_consommee[i] += E
    energie_tour[i]      += E
    tour_en_cours[i]     += 1
    positions_drones[i]   = nouveau_p
    return True


def visites_restantes(j):
    return max(visites_requises[j] - visites_effectuees[j], 0)

def score_pdp(drone_pos, j, frame, indices_actifs):
    if visites_effectuees[j] >= visites_requises[j]:
        return -np.inf

    SCALE = 1e9
    reste = visites_restantes(j)

    t = frame - derniere_visite_cibles[j]
    if not np.isfinite(t):
        t = frame + 1

    dist_actuelle = np.linalg.norm(drone_pos - cibles[j])
    autres_actifs = [k for k in indices_actifs if k != j]

    if not autres_actifs:
        score_local = 1 / (dist_actuelle + 1e-9)
    else:
        score_local = (t + 1) / (dist_actuelle + 1e-9)

    return reste * SCALE + score_local

def choisir_cible_pdp(drone_pos, indices_cellule, frame):
    actifs = cibles_incomplete(indices_cellule)
    if not actifs:
        return -1
    best_idx, best_val = -1, -np.inf
    for j in actifs:
        val = score_pdp(drone_pos, j, frame, actifs)
        if val > best_val:
            best_val = val
            best_idx = j
    return best_idx


def couleur_cible(j):
    done = visites_effectuees[j]
    req  = visites_requises[j]

    if done >= req:
        return 'gray'

    palettes = {
        1: ['green'],
        2: ['orange', 'green'],
        3: ['red', 'orange', 'green'],
    }
    palette = palettes.get(req, ['red'] * req)
    return palette[done]


def afficher_tableau_resultats():
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.axis('off')

    energie_totale  = np.sum(energie_consommee)
    comm_totale     = np.sum(energie_comm_consommee)
    energie_globale = energie_totale + comm_totale

    entetes_drones = ["Drone", "E. mouvement (J)", "E. comm (J)", "E. totale (J)"]
    donnees_drones = []
    for i in range(n_drones):
        e_tot = int(energie_consommee[i] + energie_comm_consommee[i])
        donnees_drones.append([
            f"Drone {i}",
            int(energie_consommee[i]),
            int(energie_comm_consommee[i]),
            e_tot,
        ])

    ax2.set_title("Résultats de la mission", fontsize=14, fontweight='bold', pad=20)

    t2 = ax2.table(cellText=donnees_drones, colLabels=entetes_drones,
                   loc='center', cellLoc='center')
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1, 2)

    for k in range(len(entetes_drones)):
        t2[0, k].set_facecolor('#ED7D31')
        t2[0, k].set_text_props(color='white', fontweight='bold')

    ax2.text(0.5, 0.25, f"E. totale globale : {int(energie_globale)} J",
             transform=ax2.transAxes, fontsize=11, ha='center',
             color='darkblue', fontweight='bold')

    plt.tight_layout()
    plt.show()


def ajouter_nouvelle_cible():
    global cibles, poids_cibles, couleurs_cibles, visites_requises, \
           visites_effectuees, derniere_visite_cibles, n_cibles, nb_cibles_ajoutees

    nouvelle_cible = np.random.uniform([xmin, ymin], [xmax, ymax])
    nouveau_poids  = np.random.choice([1, 2, 3])

    cibles                 = np.vstack([cibles, nouvelle_cible])
    poids_cibles           = np.append(poids_cibles, float(nouveau_poids))
    visites_requises       = np.append(visites_requises, nouveau_poids)
    visites_effectuees     = np.append(visites_effectuees, 0)
    derniere_visite_cibles = np.append(derniere_visite_cibles, -np.inf)

    if nouveau_poids == 1:
        couleur = 'green'
    elif nouveau_poids == 2:
        couleur = 'orange'
    else:
        couleur = 'red'

    couleurs_cibles    = np.append(couleurs_cibles, couleur)
    n_cibles          += 1
    nb_cibles_ajoutees += 1


def update(frame):
    global positions_drones, energie_drones, cible_courante, \
           drone_mission_terminee, temps, n_cibles

    temps += 1

    if temps % INTERVAL_AJOUT == 0 and nb_cibles_ajoutees < MAX_NOUVELLES_CIBLES:
        ajouter_nouvelle_cible()

    ax.clear()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    zx, zy = zone_global.exterior.xy
    ax.plot(zx, zy)

    for i, drone in enumerate(drones.geoms):

        if frame % comm_interval == 0:
            E_comm = calcul_energie_comm(positions_drones[i])
            energie_drones[i]         -= E_comm
            energie_comm_consommee[i] += E_comm

        if energie_drones[i] <= SEUIL_ENERGIE_CRITIQUE or en_recharge[i]:
            if not np.allclose(positions_drones[i], station, atol=1):
                drone_en_retour[i] = True
                deplacer_si_energie(i, station)
            else:
                drone_en_retour[i] = False
                if not en_recharge[i]:
                    en_recharge[i] = True
                    cooldown[i]    = 11
                if cooldown[i] > 0:
                    cooldown[i] -= 1
                else:
                    energie_drones[i] = 200000.0
                    en_recharge[i]    = False
            continue

        drone_en_retour[i] = False

        cellule_drone = None
        for cellule in cellules_coupees:
            if cellule.contains(drone):
                cellule_drone = cellule
                break

        if cellule_drone is None:
            continue

        centroide_cellule = np.array([cellule_drone.centroid.x,
                                      cellule_drone.centroid.y])
        centroides[i] = centroide_cellule

        indices_cellule = [j for j, pos in enumerate(cibles)
                           if cellule_drone.contains(Point(pos))]

        cellule_complete = (not indices_cellule) or toutes_cibles_completes_cellule(indices_cellule)

        if cellule_complete and indices_cellule:
            drone_en_attente_centroide[i] = False

            historique_temps[i].append(tour_en_cours[i])
            historique_energie[i].append(energie_tour[i])

            tour_en_cours[i] = 0
            energie_tour[i]  = 0

            for j in indices_cellule:
                visites_effectuees[j]     = 0
                derniere_visite_cibles[j] = -np.inf
            cible_courante[i] = -1
            continue
        if not indices_cellule:
            deplacer_si_energie(i, centroide_cellule)
            continue

        j_courant = cible_courante[i]
        visite_ce_frame = (frame_visite_drone[i] == frame)

        if not visite_ce_frame and (
                j_courant == -1 or
                j_courant not in indices_cellule or
                visites_effectuees[j_courant] >= visites_requises[j_courant]):
            j_courant         = choisir_cible_pdp(positions_drones[i], indices_cellule, frame)
            cible_courante[i] = j_courant

        if j_courant == -1:
            deplacer_si_energie(i, centroide_cellule)
            continue

        cible_target = cibles[j_courant]
        distance     = np.linalg.norm(cible_target - positions_drones[i])

        if distance <= 1:
            if frame_visite_drone[i] != frame:
                visites_effectuees[j_courant]    += 1
                derniere_visite_cibles[j_courant] = frame
                frame_visite_drone[i]             = frame
                cible_courante[i]                 = -1
            j_next = choisir_cible_pdp(positions_drones[i], indices_cellule, frame)
            if j_next != -1 and j_next != j_courant:
                deplacer_si_energie(i, cibles[j_next])
            else:
                deplacer_si_energie(i, centroide_cellule)
        else:
            frame_visite_drone[i] = -1
            deplacer_si_energie(i, cible_target)

    # ── Affichage simulation ───────────────────────────────────────────────────
    for cellule in cellules_coupees:
        x, y = cellule.exterior.xy
        ax.fill(x, y, color='lightgray', edgecolor='black', linewidth=0.5)

    ax.scatter(*station, color='purple', s=80, marker='^', zorder=5)
    ax.text(station[0], station[1] + 2, "Station", fontsize=7,
            ha='center', color='purple')

    for i, pos in enumerate(positions_drones):
        ax.plot([pos[0], station[0]], [pos[1], station[1]],
                color='purple', linewidth=0.3, alpha=0.4, linestyle='--')

    # ── Dessin des drones selon leur état ─────────────────────────────────────
    for i, pos in enumerate(positions_drones):
        if en_recharge[i]:
            draw_drone(ax, pos[0], pos[1], size=3.5, body_color='cyan', arm_color='teal')
            ax.text(pos[0], pos[1] + 2, f"⚡{cooldown[i]}", fontsize=7,
                    color='teal', ha='center', fontweight='bold')
        elif drone_en_retour[i]:
            draw_drone(ax, pos[0], pos[1], size=3.5, body_color='deepskyblue', arm_color='deepskyblue')
            ax.text(pos[0], pos[1] + 2, "→ST", fontsize=6,
                    color='deepskyblue', ha='center', fontweight='bold')
        elif drone_en_attente_centroide[i]:
            draw_drone(ax, pos[0], pos[1], size=3.5, body_color='orange', arm_color='orange')
            ax.text(pos[0], pos[1] + 2, "→C", fontsize=6, color='orange',
                    ha='center', fontweight='bold')
        else:
            draw_drone(ax, pos[0], pos[1], size=3.5, body_color='black', arm_color='black')

    for j in range(n_cibles):
        ax.scatter(cibles[j, 0], cibles[j, 1],
                   c=couleur_cible(j), s=20, marker='x', zorder=4)
        done = visites_effectuees[j]
        req  = visites_requises[j]
        ax.text(cibles[j, 0], cibles[j, 1] + 1.5,
                f"{done}/{req}", fontsize=6, ha='center',
                color='gray' if done >= req else 'black')

    for i, pos in enumerate(positions_drones):
        ax.text(pos[0], pos[1] + 4, f"E:{int(energie_drones[i])}",          fontsize=7)
        ax.text(pos[0], pos[1] + 6, f"Mv:{int(energie_consommee[i])}",      fontsize=7, color='blue')
        ax.text(pos[0], pos[1] + 8, f"Cm:{int(energie_comm_consommee[i])}", fontsize=7, color='purple')

    energie_totale  = np.sum(energie_consommee)
    comm_totale     = np.sum(energie_comm_consommee)
    energie_globale = energie_totale + comm_totale

    ax.text(0.02, 1.16, f"E mouvement : {int(energie_totale)} J",
            transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.02, 1.12, f"E comm      : {int(comm_totale)} J",
            transform=ax.transAxes, fontsize=10, color='purple')
    ax.text(0.02, 1.08, f"E totale    : {int(energie_globale)} J",
            transform=ax.transAxes, fontsize=10, color='red')

    ax_table.clear()
    ax_table.axis('off')

    table_data           = []
    temps_premier_tour   = []
    energie_premier_tour = []

    for i in range(n_drones):
        if len(historique_temps[i]) > 0:
            temps_tour1   = historique_temps[i][0]
            energie_tour1 = historique_energie[i][0]
        else:
            temps_tour1   = 0
            energie_tour1 = 0

        temps_premier_tour.append(temps_tour1)
        energie_premier_tour.append(energie_tour1)

        table_data.append([
            f"D{i+1}",
            f"{temps_tour1:.0f}",
            f"{energie_tour1:.1f}"
        ])

    table_data.append(["", "", ""])
    table_data.append([
        "TOTAL",
        f"{sum(temps_premier_tour):.0f}",
        f"{sum(energie_premier_tour):.1f}"
    ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Drone", "Temps Tour", "Energie Tour"],
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    total_row_index = n_drones + 1
    for j in range(3):
        cell = table[(total_row_index, j)]
        cell.set_facecolor('#D3D3D3')


ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.tight_layout()
plt.show()
