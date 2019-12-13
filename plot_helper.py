import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# taken from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(ax, mu, cov, n_std=1, facecolor="none", edgecolor="red", alpha=0.5):
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

def plot_2d_representation(model,
        data,
        path,
        title="",
        cm=plt.get_cmap('gist_rainbow'),
    ):

    images, labels = data
    q_zgx = model.encode(images)

    mu = q_zgx.mean().numpy()
    cov = q_zgx.covariance().numpy()

    plt.figure()
    ax = plt.gca()

    if title:
        plt.title(title)

    for i in range(10):
        ix = np.argwhere(labels == i)
        ix = ix.reshape(-1)
        plt.scatter(mu[ix, 0], mu[ix, 1], color=cm(i/10), alpha=0.0, label=i)

        for j in range(ix.shape[0]):
            ij = ix[j]
            plot_ellipse(ax, mu[ij,:], cov[ij, :, :], edgecolor=cm(i/10))

    plt.savefig(path)
    plt.close("all")

    z = np.concatenate([labels.reshape((-1, 1)), mu, cov.reshape(cov.shape[0], -1)], axis=1)

    np.save(path.replace(".png", ""), z)