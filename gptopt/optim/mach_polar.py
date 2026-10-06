from itertools import repeat
import torch

coeffs_list = [
    [-2.1450087827058155, 0.29396535045034056, 0.16564530452274018, 0.99889294797206141, -0.90666574610969097, 1.9988929479720614, 25.99136111644188, 12052.603744244119, -11279.144155547205, 48195.375794393949],
    [-8.3584841158700911, 10.386880488053521, -19.92000744623607, 22.259751178598798, -23.925543205546319, 23.259751178598798, 11.378064053335891, 9.5611937774860962, -2.4307134281540357, 0.39687811202674833],
    [-6.5368915076838654, 6.6922403382938569, -10.8464548186179, 13.861719625870077, -13.265513014737223, 14.861719625870077, 4.8174159867530744, 1.1175961931200895, -0.6776202475841302, 0.25972273106529664],
    [-4.5367249397028342, 5.7643010237648298, -6.568655402755553, 7.5420861565344683, -7.1722962015345217, 8.5420861565344683, 3.3386866475066785, -0.80683498876942927, -0.17342231003879149, 0.19584143052638475],
]

# safety factor for numerical stability on f0, f1, f2, f4 (but exclude last polynomial)
coeffs_list = [
    tuple(c[:6]) + tuple(f / 1.01 for f in c[6:]) if i < len(coeffs_list) - 1 else tuple(c)
    for i, c in enumerate(coeffs_list)
]


@torch.compile
def MachPolar17(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    hs = coeffs_list[:steps] + list(
        repeat(coeffs_list[-1], steps - len(coeffs_list)))

    for c1, d0, d1, d2, e1, e2, f0, f1, f2, f4 in hs:
        S1 = X @ X.mT
        S0 = torch.eye(S1.size(-1), dtype=S1.dtype, device=S1.device).expand_as(S1)
        S2 = S1 @ S1
        S3 = S2 @ (c1 * S1 + S2)
        S4 = (d0 * S0 + d1 * S1 + d2 * S2 + S3) @ (e1 * S1 + e2 * S2 + S3)
        X = (f0 * S0 + f1 * S1 + f2 * S2 + f4 * S4) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X