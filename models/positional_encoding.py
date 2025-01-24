import math

import torch
from torch import nn

from .spherical_armonics import SH as SH_analytic


class SphericalHarmonics(nn.Module):
    """
    Spherical Harmonics locaiton encoder
    """

    def __init__(self, legendre_polys: int = 10, harmonics_calculation="analytic"):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        if harmonics_calculation == "closed-form":
            self.SH = SH_closed_form
        elif harmonics_calculation == "analytic":
            self.SH = SH_analytic

    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)
        """
        greater_than_50 = (lon > 50).any() or (lat > 50).any()
        if greater_than_50:
            SH = SH_closed_form
        else:
            SH = SH_analytic
        """
        SH = self.SH

        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                if y.isnan().any():
                    print(m, l, y)
                Y.append(y)

        return torch.stack(Y, dim=-1)


####################### Spherical Harmonics utilities ########################
# Code copied from https://github.com/BachiLi/redner/blob/master/pyredner/utils.py
# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def SH_renormalization(l, m):
    return math.sqrt(
        (2.0 * l + 1.0) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m))
    )


def SH_closed_form(m, l, phi, theta):
    if m == 0:
        return SH_renormalization(l, m) * associated_legendre_polynomial(
            l, m, torch.cos(theta)
        )
    elif m > 0:
        return (
            math.sqrt(2.0)
            * SH_renormalization(l, m)
            * torch.cos(m * phi)
            * associated_legendre_polynomial(l, m, torch.cos(theta))
        )
    else:
        return (
            math.sqrt(2.0)
            * SH_renormalization(l, -m)
            * torch.sin(-m * phi)
            * associated_legendre_polynomial(l, -m, torch.cos(theta))
        )
