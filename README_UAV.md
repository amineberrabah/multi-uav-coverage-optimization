# 🚁 Multi-UAV Coverage Optimization

**Deployment Optimization and Collaborative Coverage of a Multi-UAV System Using Lloyd and CNN Approaches**

> Master's Thesis — Université Abou Bekr Belkaïd de Tlemcen  
> Réseaux et Systèmes Distribués — 2026

---

## 📌 Overview

This project implements and compares two approaches for optimizing the deployment and collaborative coverage of a multi-UAV (Unmanned Aerial Vehicle) system:

- **Lloyd's Algorithm** — geometric Voronoï-based coverage optimization
- **CNN Actor-Critic** — deep learning-based coverage using Convolutional Neural Networks

---

## 📁 Project Structure

```
├── LloydPur100.py                          # Lloyd algorithm — 100×100 zone
├── LloydPur1000.py                         # Lloyd algorithm — 1000×1000 zone
├── CNN1000.py                              # CNN Actor-Critic — 1000×1000 zone
├── Voronoi_centroideCNN_ajoutdroneCNN100.py # Voronoï + CNN centroid with drone addition
├── méthode_PD.py                           # Priority-Driven Patrolling (PDP)
├── méthode_pdp.py                          # PDP scoring method
├── pve.py                                  # PVE — Elastic Visit Potential (Gaussian fields)
├── change position.TXT                     # Position change logs
├── cibles.npy                              # Target positions (NumPy)
├── couleurs_cibles.npy                     # Target colors (NumPy)
└── poids_cibles.npy                        # Target weights (NumPy)
```

---

## 🛠 Technologies

- **Python 3**
- **NumPy** — numerical computation
- **Matplotlib** — visualization & animation
- **Shapely** — geometric operations
- **PyTorch / TensorFlow** — CNN Actor-Critic models

---

## 🚀 How to Run

```bash
# Install dependencies
pip install numpy matplotlib shapely

# Run Lloyd algorithm (100x100)
python LloydPur100.py

# Run CNN approach (1000x1000)
python CNN1000.py

# Run PVE method
python pve.py
```

---

## 📊 Methods

### Lloyd's Algorithm
Iterative Voronoï-based partitioning that moves UAVs toward the centroid of their coverage zone until convergence.

### CNN Actor-Critic
Deep reinforcement learning approach where UAVs learn optimal coverage policies using a Convolutional Neural Network.

### PDP (Priority-Driven Patrolling)
Scoring-based method that prioritizes zones based on visit urgency and importance weights.

### PVE (Potentiel de Visite Élastique)
Gaussian potential field approach that guides UAVs toward uncovered zones using elastic visit potentials.

---

## 👨‍💻 Author

**Mohamed Amine BERRABAH** — Master 2, Réseaux et Systèmes Distribués  
Université Abou Bekr Belkaïd de Tlemcen, Algeria — 2026

---

## 📄 License

Academic project — All rights reserved.
