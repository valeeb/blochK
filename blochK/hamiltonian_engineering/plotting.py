import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as tck
from jax import numpy as jnp

def plot_3d_d_matrix(kx, ky, kz, d_function):
    k_xyz_vals = np.array(np.meshgrid(kx, ky, kz))
    d_values = d_function(k_xyz_vals)
    xy_mags = np.sqrt(d_values[1] ** 2 + d_values[2] ** 2)

    # magnitude and colormap
    d_mag = d_values[0] + np.sqrt(jnp.einsum("i...,i...->...", d_values[1:], d_values[1:]))
    cvals = [0.0, *np.linspace(0.05 * np.max(d_mag), np.max(d_mag), 10)]
    colors = [
        "red",
        *[matplotlib.colormaps["Blues"](i) for i in np.linspace(0, 1, 10)],
    ]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    document_width = 240
    figwidth = document_width / 25.4
    figheight = len(kz) * figwidth / 3 - 2
    fig, axes = plt.subplots(
        len(kz), 3, figsize=(document_width / 25.4, figheight), dpi=200
    )
    for j, _ in enumerate(kz):

        # xy_part
        KX = k_xyz_vals[0, :, :, j]/np.pi
        KY = k_xyz_vals[1, :, :, j]/np.pi
        axes[j, 0].streamplot(
            KX,
            KY,
            d_values[1, :, :, j],
            d_values[2, :, :, j],
            color="black",
            linewidth=0.5,
            density=1.3,
            arrowsize=0.4,
        )
        im_mag = axes[j, 0].pcolor(
            KX,
            KY,
            xy_mags[:, :, j],
            cmap="Greens",
            vmin=0,
            vmax=np.max(xy_mags),
        )

        # z_part
        mag_z = axes[j, 1].pcolor(
            KX,
            KY,
            d_values[3, :, :, j],
            cmap="RdBu_r",
            vmin=-np.max(np.abs(d_values[3, :, :, :])),
            vmax=np.max(np.abs(d_values[3, :, :, :])),
        )

        mag_sum = axes[j, 2].pcolor(
            KX,
            KY,
            d_mag[:, :, j],
            cmap=cmap,
            norm=norm,
        )

        # choose colourmap for cbar
        cbar1 = fig.colorbar(im_mag, ax=axes[j, 0], fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(mag_z, ax=axes[j, 1], fraction=0.046, pad=0.04)
        cbar3 = fig.colorbar(mag_sum, ax=axes[j, 2], fraction=0.046, pad=0.04)
        axes[j, 0].set_title(r"$xy$ Components, $k_z={:.2f}\pi$".format(kz[j] / np.pi))
        axes[j, 1].set_title(r"$z$ Component, $k_z={:.2f}\pi$".format(kz[j] / np.pi))
        axes[j, 2].set_title(r"Total Magnitude, $k_z={:.2f}\pi$".format(kz[j] / np.pi))

    for a in axes.flatten():
        a.set_aspect("equal")
        a.axhline(1, linestyle="--", c="k",linewidth = 0.8)
        a.axhline(-1, linestyle="--", c="k",linewidth = 0.8)
        a.axvline(1, linestyle="--", c="k",linewidth = 0.8)
        a.axvline(-1, linestyle="--", c="k",linewidth = 0.8)

        a.xaxis.set_major_formatter(tck.FormatStrFormatter(r"%g $\pi$"))
        a.xaxis.set_major_locator(tck.MultipleLocator(base=.5))
        a.yaxis.set_major_formatter(tck.FormatStrFormatter(r"%g $\pi$"))
        a.yaxis.set_major_locator(tck.MultipleLocator(base=.5))

    return fig, axes
