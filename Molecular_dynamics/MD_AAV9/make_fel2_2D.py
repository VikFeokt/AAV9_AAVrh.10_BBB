import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage as ndimage

# =====================
# Чтение PCA
# =====================
data = np.loadtxt('pca.dat', skiprows=1)

pc1 = data[:, 1]
pc2 = data[:, 2]

# =====================
# Расчёт 2D free energy landscape
# =====================
def calculate_free_energy_2d(x, y, temp=310.15, nbins=200, sigma=2.0):

    hist, xedges, yedges = np.histogram2d(
        x, y,
        bins=nbins,
        density=True
    )

    # Избегаем log(0)
    hist = np.where(hist > 0, hist, 1e-10)

    # R в cal/(mol*K)
    R = 1.987

    # Free energy в kcal/mol
    free_energy = -R * temp * np.log(hist) / 1000

    # Переносим минимум в 0
    free_energy -= np.min(free_energy)

    # Центры bins
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    X, Y = np.meshgrid(x_centers, y_centers)

    # Сглаживание
    F = ndimage.gaussian_filter(
        free_energy.T,
        sigma=sigma
    )

    # После сглаживания снова делаем минимум = 0
    F -= np.min(F)

    return X, Y, F


X, Y, F = calculate_free_energy_2d(
    pc1,
    pc2,
    temp=310.15,
    nbins=200,
    sigma=2.0
)

# =====================
# Цветовая схема
# =====================
colors = [
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0)
]

custom_cmap = LinearSegmentedColormap.from_list(
    'blue_to_red',
    colors,
    N=256
)

# =====================
# Ограничение энергии
# =====================
ENERGY_LIMIT = 5

F_plot = np.clip(F, 0, ENERGY_LIMIT)

# =====================
# Построение 2D FEL
# =====================
fig, ax = plt.subplots(
    figsize=(10, 8),
    dpi=1200
)

contour = ax.contourf(
    X,
    Y,
    F_plot,
    levels=100,
    cmap=custom_cmap,
    vmin=0,
    vmax=ENERGY_LIMIT
)

# Контурные линии
lines = ax.contour(
    X,
    Y,
    F_plot,
    levels=10,
    colors='black',
    linewidths=0.4,
    alpha=0.4
)

# =====================
# Оси
# =====================
ax.set_xlabel(
    'PC1',
    fontsize=18,
    fontweight='bold'
)

ax.set_ylabel(
    'PC2',
    fontsize=18,
    fontweight='bold'
)

ax.set_title(
    'Gibbs Energy Landscape',
    fontsize=20,
    fontweight='bold'
)

ax.tick_params(
    axis='both',
    labelsize=14
)

# =====================
# Colorbar
# =====================
cbar = plt.colorbar(
    contour,
    ax=ax,
    pad=0.03
)

cbar.set_label(
    'Free Energy (kcal/mol)',
    fontsize=16,
    fontweight='bold'
)

cbar.ax.tick_params(
    labelsize=13
)

# =====================
# Сетка
# =====================
ax.grid(
    True,
    linestyle='--',
    alpha=0.25
)

plt.tight_layout()

# =====================
# Сохранение
# =====================
plt.savefig(
    'FEL_2D.png',
    dpi=1200,
    bbox_inches='tight'
)

plt.show()

# =====================
# Информация
# =====================
print('=' * 50)
print('2D FREE ENERGY LANDSCAPE')
print('=' * 50)
print(f'Minimum free energy: {F.min():.4f} kcal/mol')
print(f'Maximum free energy: {F.max():.4f} kcal/mol')
print(f'Display limit: {ENERGY_LIMIT:.2f} kcal/mol')
