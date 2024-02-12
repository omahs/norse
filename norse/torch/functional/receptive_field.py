"""
These receptive fields are derived from scale-space theory, specifically in the paper `Normative theory of visual receptive fields by Lindeberg, 2021 <https://www.sciencedirect.com/science/article/pii/S2405844021000025>`_.

For use in spiking / binary signals, see the paper on `Translation and Scale Invariance for Event-Based Object tracking by Pedersen et al., 2023 <https://dl.acm.org/doi/10.1145/3584954.3584996>`_
"""

from typing import List, Tuple, Union, Optional

import torch
import torchvision

dymask = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
dxmask = dymask.T
dyymask = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
dxxmask = dyymask.T
dxymask = torch.tensor([[-0.25, 0, 0.25], [0, 0, 0], [0.25, 0, -0.25]])


def gaussian_kernel(x, c):
    """
    Efficiently creates a 2d gaussian kernel.

    Arguments:
      x (torch.Tensor): A 2-d matrix
      s (float): The variance of the gaussian
      c (torch.Tensor): A 2x2 covariance matrix describing the eccentricity of the gaussian
    """
    ci = torch.linalg.inv(c)
    cd = torch.linalg.det(c)
    fraction = 1 / (2 * torch.pi * torch.sqrt(cd))
    b = torch.einsum("bimj,jk->bik", -x.unsqueeze(2), ci)
    a = torch.einsum("bij,bij->bi", b, x)
    return fraction * torch.exp(a / 2)


# def covariance_matrix(angle: float, ratio: float, scale: float = 2.5):
# sm = torch.tensor([1, ratio])
# r = torch.tensor(
# [[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]],
# dtype=torch.float32,
# )
# return ((r * sm) @ (sm * r).T) * scale


def covariance_matrix(sigma1: float, sigma2: float, phi: float):
    lambda1 = torch.as_tensor(sigma1) ** 2
    lambda2 = torch.as_tensor(sigma2) ** 2
    phi = torch.as_tensor(phi)
    cxx = lambda1 * torch.cos(phi) ** 2 + lambda2 * torch.sin(phi) ** 2
    cxy = (lambda1 - lambda2) * torch.cos(phi) * torch.sin(phi)
    cyy = lambda1 * torch.sin(phi) ** 2 + lambda2 * torch.cos(phi) ** 2
    return torch.tensor([[cxx, cxy], [cxy, cyy]])


def spatial_receptive_field(
    angle, ratio, size: int, scale: float = 1, domain: float = 10
):
    """
    Creates a (size x size) receptive field kernel

    Arguments:
      angle (float): The rotation of the kernel in radians
      ratio (float): The eccentricity as a ratio
      size (int): The size of the square kernel in pixels
      scale (float): The scale of the field. Defaults to 2.5
      domain (float): The initial coordinates from which the field is sampled. Defaults to 8 (equal to -8 to 8).
    """
    angle = torch.as_tensor(angle)
    a = torch.linspace(-domain, domain, size)
    c = covariance_matrix(ratio, 1 / ratio, angle) * scale
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    coo = torch.stack([xs, ys], dim=2)
    k = gaussian_kernel(coo, c)
    return k / k.sum()

def _extract_derivatives(
    derivatives: Union[int, List[Tuple[int, int]]]
) -> Tuple[List[Tuple[int, int]], int]:
    if isinstance(derivatives, int):
        if derivatives == 0:
            return [(0, 0)], 0
        else:
            return [
                (x, y) for x in range(derivatives + 1) for y in range(derivatives + 1)
            ], derivatives
    elif isinstance(derivatives, list):
        return derivatives, max([max(x, y) for (x, y) in derivatives])
    else:
        raise ValueError(
            f"Derivatives expected either a number or a list of tuples, but got {derivatives}"
        )


def calculate_normalization(dx: int, scale: float, gamma: float = 1):
    """
    Calculates scale normalization for a spatial receptive field at a given directional derivative
    Lindeberg: Feature detection with automatic scale selection, eq. 20
    https://doi.org/10.1023/A:1008045108935

    Arguments:
        dx (int): The nth directional derivative
        scale (float): The scale of the receptive field
        gamma (float): A normalization parameter
    """
    t = scale**2
    scale_norm = scale ** (dx * (1 - gamma))
    xi_norm = t ** (gamma / 2)
    return scale_norm * xi_norm


def derive_kernel(kernel, angle):
    """
    Takes the spatial derivative at a given angle
    """
    dirx = torch.cos(angle)
    diry = torch.sin(angle)
    gradx = torch.gradient(kernel, dim=0)[0] * dirx
    grady = torch.gradient(kernel, dim=1)[0] * diry
    derived = gradx + grady
    return derived


def spatial_receptive_field(angle: float, ratio: float, size: int, domain=8, scale=8):
    """
    Creates a spatial receptive field, shapes as a size x size tensor.
    """
    angle = torch.as_tensor(angle)
    c = covariance_matrix(1 / ratio, (ratio + 1e-8), angle) * scale
    a = torch.linspace(-domain, domain, size)
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    coo = torch.stack([xs, ys], dim=2)
    g = gaussian_kernel(coo, c)
    return g


def derive_spatial_receptive_field(
    field: torch.Tensor, angle: float, scale: float, derivatives: List[Tuple[int, int]]
):
    angle = torch.as_tensor(angle)
    kernels = []
    for dx, dy in derivatives:
        derived = field
        while dx > 0 or dy > 0:
            if dx > 0:
                derived = derive_kernel(derived, angle) * calculate_normalization(
                    1, scale, 1
                )
                dx -= 1
            if dy > 0:
                derived = derive_kernel(
                    derived, angle + torch.pi / 2
                ) * calculate_normalization(1, scale, 1)
                dy -= 1
        kernels.append(derived)
    return torch.stack(kernels)


def spatial_receptive_fields_with_derivatives(
    n_scales: int,
    n_angles: int,
    n_ratios: int,
    size: int,
    derivatives: Union[int, List[Tuple[int, int]]] = 0,
    min_scale: float = 0.8,
    max_scale: float = 1.3,
    min_ratio: float = 0.8,
    max_ratio: float = 1.3,
    min_angle: float = 0,
    max_angle: Optional[float] = None,
    domain: Optional[int] = None,
) -> torch.Tensor:
    r"""
    Creates a number of receptive field with 1st directional derivatives.
    The parameters decide the number of combinations to scan over, i. e. the number of receptive fields to generate.
    Specifically, we generate ``derivatives * (n_angles * n_scales * (n_ratios - 1) + n_scales)`` fields.
    The ``(n_ratios - 1) + n_scales`` terms exist because at ``ratio = 1``, fields are perfectly symmetrical, and there
    is therefore no reason to scan over the angles and scales for ``ratio = 1``.
    However, ``n_scales`` receptive fields still need to be added (one for each scale-space).
    Finally, the ``derivatives *`` term comes from the addition of spatial derivatives.
    Arguments:
        n_scales (int): Number of scaling combinations (the size of the receptive field) drawn from a logarithmic distribution
        n_angles (int): Number of angular combinations (the orientation of the receptive field)
        n_ratios (int): Number of eccentricity combinations (how "flat" the receptive field is)
        size (int): The size of the square kernel in pixels
        derivatives (Union[int, List[Tuple[int, int]]]): The spatial derivatives to include. Defaults to 0 (no derivatives).
            Can either be a number, in which case 1 + 2 ** n derivatives will be made (except when 0, see below).
              Example: `derivatives=0` omits derivatives
              Example: `derivatives=1` provides 2 spatial derivatives + 1 without derivation
            Or a list of tuples specifying the derivatives in both spatial dimensions
              Example: `derivatives=[(0, 0), (1, 2)]` provides two outputs, one without derivation and one :math:`\partial_x \partial^2_y`
    """
    if max_angle is None:
        max_angle = torch.pi - torch.pi / n_angles
    angles = torch.linspace(min_angle, max_angle, n_angles)
    ratios = torch.linspace(min_ratio, max_ratio, n_ratios)
    scales = torch.exp(torch.linspace(min_scale, max_scale, n_scales))
    derivatives, max_deriv = _extract_derivatives(derivatives)
    if domain is None:
        domain = size // 2
    fields = []

    # We add extra space in both the domain and size to account for the derivatives
    for scale in scales:
        for ratio in ratios:
            if ratio == 1:
                field = spatial_receptive_field(
                    angle, 0, size, scale=scale, domain=domain
                )
                derived = derive_spatial_receptive_field(
                    field, angle, scale, derivatives=derivatives
                )
                fields.extend(derived)
            else:
                for angle in angles:
                    field = spatial_receptive_field(
                        angle, ratio, size, scale=scale, domain=domain
                    )
                    derived = derive_spatial_receptive_field(
                        field, angle, scale=scale, derivatives=derivatives
                    )
                    fields.extend(derived)
    return torch.stack(fields)


def temporal_scale_distribution(
    n_scales: int,
    min_scale: float = 1,
    max_scale: Optional[float] = None,
    c: Optional[float] = 1.41421,
):
    r"""
    Provides temporal scales according to [Lindeberg2016].
    The scales will be logarithmic by default, but can be changed by providing other values for c.

    .. math:
        \tau_k = c^{2(k - K)} \tau_{max}
        \mu_k = \sqrt(\tau_k - \tau_{k - 1})

    Arguments:
      n_scales (int): Number of scales to generate
      min_scale (float): The minimum scale
      max_scale (Optional[float]): The maximum scale. Defaults to None. If set, c is ignored.
      c (Optional[float]): The base from which to generate scale values. Should be a value
        between 1 to 2, exclusive. Defaults to sqrt(2). Ignored if max_scale is set.

    .. [Lindeberg2016] Lindeberg 2016, Time-Causal and Time-Recursive Spatio-Temporal
        Receptive Fields, https://link.springer.com/article/10.1007/s10851-015-0613-9.
    """
    xs = torch.linspace(1, n_scales, n_scales)
    if max_scale is not None:
        if n_scales > 1:  # Avoid division by zero when having a single scale
            c = (min_scale / max_scale) ** (1 / (2 * (n_scales - 1)))
        else:
            return torch.tensor([min_scale]).sqrt()
    else:
        max_scale = (c ** (2 * (n_scales - 1))) * min_scale
    taus = c ** (2 * (xs - n_scales)) * max_scale
    return taus.sqrt()
