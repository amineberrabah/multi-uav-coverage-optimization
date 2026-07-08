# ============================================================
#  Simulation de couverture multi-drones par tessellation de Voronoï
#  Objectif : minimiser un coût (couverture + variance) en ajoutant
#  dynamiquement des drones jusqu'à atteindre un seuil prédéfini.
# ============================================================

from shapely import voronoi_polygons, MultiPoint
from shapely.geometry import box, Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Initialisation des paramètres globaux ──────────────────────────────────────
np.random.seed(44)
R = 20  # Rayon de détection de chaque drone
n_drones = 2  # Nombre initial de drones
n_cibles = 100  # Nombre total de cibles à couvrir

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

# Pondération du coût : alpha pour la couverture, beta pour la variance
alpha = 0.6
beta = 0.4
couts = []

# Limites de la zone de simulation
xmin, xmax = 0, 100
ymin, ymax = 0, 100

# Tableaux des positions et des centroïdes calculés
centroides = np.zeros((n_drones, 2))
positions_drones = np.random.uniform([xmin, ymin], [xmax, ymax], (n_drones, 2))
cibles = np.random.uniform([xmin, ymin], [xmax, ymax], (n_cibles, 2))

variancee = 0

# ── Répartition des cibles en 3 niveaux de priorité (poids 3, 2, 1) ───────────
n1 = int(0.33 * n_cibles)
n2 = int(0.33 * n_cibles)
n3 = n_cibles - n1 - n2

poids_cibles = np.array([3] * n1 + [2] * n2 + [1] * n3)
np.random.shuffle(poids_cibles)

# Couleurs associées aux priorités pour l'affichage
couleurs_cibles = []
for p in poids_cibles:
    if p == 3:
        couleurs_cibles.append("red")
    elif p == 2:
        couleurs_cibles.append("orange")
    else:
        couleurs_cibles.append("green")
couleurs_cibles = np.array(couleurs_cibles)

# Polygone représentant la zone globale de simulation
zone_global = box(xmin, ymin, xmax, ymax)

# ── Fonctions utilitaires ──────────────────────────────────────────────────────


def calcul_taux_couverture(idx):
    # Calcule le pourcentage de poids couvert par un drone dans sa cellule
    somme_poids_total = np.sum(poids_cibles)
    somme_poids_cellule = 0
    for i in idx:
        somme_poids_cellule += poids_cibles[i]
    return (somme_poids_cellule / somme_poids_total) * 100


def calcul_variance(cellules, drones):
    # Calcule la variance du nombre de cibles couvertes entre les drones
    C = []
    for i, drone in enumerate(drones.geoms):
        for cellule in cellules:
            if cellule.contains(drone):
                c_i = 0
                for j, cible in enumerate(cibles) :
                    if cellule.contains(Point(cible)):
                        c_i += 1
                C.append(c_i)
                break
    C = np.array(C)
    if len(C) == 0:
        return 0, C
    c_bar = np.mean(C)
    variance = np.mean((C - c_bar) ** 2)
    return variance, C


def deplacement(p_drone, p_dist):
    # Déplace le drone d'un pas unitaire vers sa destination (centroïde)
    dir = p_dist - p_drone
    distance = np.linalg.norm(dir)
    if distance < 1:
        return p_dist
    else:
        dir = dir / distance
        return p_drone + dir 


def centroid(cibles_cellule, cellule):
    # Calcule le centroïde des cibles dans la cellule (ou le centre géométrique si vide)
    if len(cibles_cellule) == 0:
        return [cellule.centroid.x, cellule.centroid.y]
    cibles_cellule = np.array(cibles_cellule)
    x = np.mean(cibles_cellule[:, 0])
    y = np.mean(cibles_cellule[:, 1])
    return np.array([x, y])


n_drones_max = 1000
ani = None


def calcul_seuil_initial(xmin, xmax, ymin, ymax, n_cibles, R, alpha, beta):
    # Calcule un seuil de coût théorique basé sur la densité et la taille de la zone
    surface_zone = (xmax - xmin) * (ymax - ymin)
    densite = n_cibles / surface_zone
    taux_attendu = min(1.0, densite * np.pi * R**2)
    variance_ref = densite * R**2
    return alpha * (1.0 - taux_attendu) + beta * variance_ref


# Calcul du seuil fixe de coût acceptable (ne change pas au cours de la simulation)
COUT_SEUIL = calcul_seuil_initial(xmin, xmax, ymin, ymax, n_cibles, R, alpha, beta)
surface_zone_init = (xmax - xmin) * (ymax - ymin)
densite_init = n_cibles / surface_zone_init
taux_att_init = min(1.0, densite_init * np.pi * R**2)
print(f"[Init] Seuil de coût prévisionnel (FIXE) : {COUT_SEUIL:.4f}")
print(
    f"       zone={xmax}×{ymax}  n_cibles={n_cibles}  R={R}"
    f"  densité={densite_init:.4f}/u²  taux_attendu={taux_att_init:.2%}"
)

# ════════════════════════════════════════════════
#  FIGURE — 2 sous-graphes : simulation + tableau
# ════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1)  # Fenêtre de simulation
ax_table = fig.add_subplot(1, 2, 2)  # Tableau d'historique des convergences
ax_table.axis("off")

# Liste mémorisant chaque convergence atteinte au fil des frames
historique_convergences = []


def update(frame):
    # ── Déclaration des variables globales modifiées à chaque frame ──────────
    global ani, couts, variancee, centroides, positions_drones, cibles, poids_cibles, couleurs_cibles, vitesse, n_drones

    ax.clear()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # Tracé du contour de la zone globale
    zx, zy = zone_global.exterior.xy
    ax.plot(zx, zy)

    # Construction de la tessellation de Voronoï à partir des positions des drones
    drones = MultiPoint(positions_drones)

    cellules = voronoi_polygons(drones, extend_to=zone_global)
    cellules_coupees = [cellule.intersection(zone_global) for cellule in cellules.geoms]

    # Calcul de l'enveloppe convexe des cibles pour restreindre les déplacements
    polygon_cibles = MultiPoint(cibles).convex_hull
    centroide_poly_cibles = [polygon_cibles.centroid.x, polygon_cibles.centroid.y]
    cellules_coupees_polygon_cibles = [
        cellule.intersection(polygon_cibles) for cellule in cellules_coupees
    ]

    # ── Mise à jour des centroïdes et déplacement de chaque drone ────────────
    taux_couverture_drone = np.zeros(n_drones)
    for i, drone in enumerate(drones.geoms):
        cercle = drone.buffer(R)
        a_une_cible = any(cercle.covers(Point(cible)) for cible in cibles)
        if not polygon_cibles.covers(drone) and not a_une_cible:
            # Si le drone est hors de la zone des cibles, il se dirige vers le centre de cette zone
            centroides[i] = centroide_poly_cibles
        else:
            for cellule in cellules_coupees_polygon_cibles:
                if cellule.covers(drone):
                    idx = []

                    cibles_cellule = []
                    # Collecte des cibles dans la cellule ET dans le rayon du drone
                    for j, cible in enumerate(cibles):
                        if cellule.covers(Point(cible)) and cercle.covers(Point(cible)):
                            idx.append(j)
                            cibles_cellule.append(cible)
                    if len(idx) > 0:
                        taux_couverture_drone[i] = calcul_taux_couverture(idx)
                    centroides[i] = centroid(cibles_cellule, cellule)
                    break

        # Déplacement d'un pas vers le centroïde calculé
        positions_drones[i] = deplacement(positions_drones[i], centroides[i])

    # ════ AFFICHAGE SIMULATION ════
    # Remplissage des cellules de Voronoï en gris clair
    for cellule in cellules_coupees:
        x, y = cellule.exterior.xy
        ax.fill(x, y, color="lightgray", edgecolor="black", linewidth=0.5)

    for pos in positions_drones:
        draw_drone(ax, pos[0], pos[1], size=3.5)
    ax.plot([], [], color='black', marker='o', markersize=6,
            linestyle='None', label='Drones')   # entrée légende

    ax.scatter(centroides[:, 0], centroides[:, 1], color="red", s=10, marker="o")

    # Tracé du contour du cercle de détection de chaque drone (intersection avec sa cellule)
    for drone in drones.geoms:
        cercle = drone.buffer(R)
        for cellule in cellules_coupees:
            if cellule.covers(drone):
                partie = cercle.intersection(cellule)
                if not partie.is_empty:
                    if partie.geom_type == "Polygon":
                        x, y = partie.exterior.xy
                        ax.plot(x, y, color="black", linewidth=0.5)
                    elif partie.geom_type == "MultiPolygon":
                        for poly in partie.geoms:
                            x, y = poly.exterior.xy
                            ax.plot(x, y, color="black", linewidth=0.5)

    # Affichage des drones, des cibles et des centroïdes

    ax.scatter(cibles[:, 0], cibles[:, 1], c=couleurs_cibles, s=30, marker="x")
    ax.scatter(centroides[:, 0], centroides[:, 1], color="red", s=10, marker="o")

    # Vérifie si tous les drones ont atteint leur centroïde (convergence)
    convergence = np.all(positions_drones == centroides)

    # Calcul de la variance et du coût global à cette frame
    variancee, coutVar = calcul_variance(cellules_coupees, drones)
    taux_couverture_drones = np.sum(taux_couverture_drone)
    cout = alpha * (1 - taux_couverture_drones / 100) + beta * variancee
    if n_drones == n_drones_max and convergence:
        ani.event_source.stop()
    # ════ HISTORIQUE DES CONVERGENCES ════
    if convergence:
        entree = {
            "frame": frame,
            "n_drones": n_drones,
            "taux": taux_couverture_drones,
            "variance": variancee,
            "cout": cout,
            "seuil": COUT_SEUIL,
        }
        if not historique_convergences or historique_convergences[-1]["frame"] != frame:
            historique_convergences.append(entree)

    # ── Arrêt si le coût est en-dessous du seuil (objectif atteint) ──────────
    if convergence and cout <= COUT_SEUIL :
        print(positions_drones)
        ani.event_source.stop()

    # ── Ajout d'un drone si convergence sans satisfaire le seuil ─────────────
    if convergence and cout > COUT_SEUIL :
        # Identifie le drone avec le plus de cibles non couvertes
        idx_max = np.argmax(coutVar)
        drone_cible = list(drones.geoms)[idx_max]

        # Localise les cibles hors du rayon du drone le moins performant
        points_non_couverts = []
        for cellule in cellules_coupees:
            if cellule.covers(drone_cible):
                cercle = drone_cible.buffer(R)
                for j, cible in enumerate(cibles):
                    p = Point(cible)
                    if cellule.covers(p) and not cercle.covers(p):
                        points_non_couverts.append(cible)
                break

        # Place le nouveau drone au centre des points non couverts
        if len(points_non_couverts) > 0:
            points_non_couverts = np.array(points_non_couverts)
            nouveau_drone = np.mean(points_non_couverts, axis=0)
        else:
            # Décalage minimal si aucun point non couvert n'est trouvé
            nouveau_drone = [drone_cible.x + 1, drone_cible.y]

        # Ajout du nouveau drone aux tableaux globaux
        positions_drones = np.vstack([positions_drones, nouveau_drone])
        centroides = np.vstack([centroides, [0, 0]])
        n_drones = len(positions_drones)

    # Étiquettes textuelles : taux de couverture et coût par drone
    for i, pos in enumerate(positions_drones):
        if i < len(coutVar):
            txt = f"T:{taux_couverture_drone[i]:.1f}%\nC:{coutVar[i]}"
            ax.text(pos[0], pos[1] + 4, txt, fontsize=9, color="blue")

    # Statistiques globales affichées au-dessus du graphe
    ax.text(
        0.02,
        1.05,
        f"Couverture: {taux_couverture_drones:.2f}%",
        transform=ax.transAxes,
        color="red",
    )
    ax.text(
        0.5, 1.05, f"Variance: {variancee:.2f}", transform=ax.transAxes, color="red"
    )

    # ════ TABLEAU CONVERGENCES ════
    ax_table.clear()
    ax_table.axis("off")
    titre = "Historique des convergences"
    if n_drones >= n_drones_max:
        titre += f"  : MAX {n_drones_max} drones atteint"
    ax_table.set_title(
        titre,
        fontsize=10,
        fontweight="bold",
        color="red" if n_drones >= n_drones_max else "black",
    )

    # Construction du tableau si au moins une convergence a eu lieu
    if historique_convergences:
        col_labels = ["#", "Frame", "Drones", "Taux", "Variance", "Coût", "Seuil"]
        rows = []
        for i, e in enumerate(historique_convergences):
            rows.append(
                [
                    str(i + 1),
                    str(e["frame"]),
                    str(e["n_drones"]),
                    f"{e['taux']:.1f}%",
                    f"{e['variance']:.3f}",
                    f"{e['cout']:.4f}",
                    f"{e['seuil']:.4f}",
                ]
            )
        # Mise en évidence de la ligne avec le coût minimal
        cout_min = min(e["cout"] for e in historique_convergences)
        cell_colors = [
            ["#d4f7d4"] * 7 if e["cout"] == cout_min else ["#ffffff"] * 7
            for e in historique_convergences
        ]
        table = ax_table.table(
            cellText=rows,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.1, 1.5)
        ax_table.text(
            0.5,
            0.02,
            f"Ligne verte = meilleur coût  |  Seuil fixe = {COUT_SEUIL:.4f}",
            ha="center",
            va="bottom",
            transform=ax_table.transAxes,
            fontsize=7,
            color="green",
        )
    else:
        # Message d'attente si aucune convergence n'a encore eu lieu
        ax_table.text(
            0.5,
            0.5,
            "En attente de la\npremière convergence...",
            ha="center",
            va="center",
            transform=ax_table.transAxes,
            fontsize=10,
            color="gray",
        )

    fig.tight_layout()


# ── Lancement de l'animation (50 ms par frame, sans mise en cache) ────────────
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.tight_layout()
plt.show()
