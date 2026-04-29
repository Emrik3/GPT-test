import os
import uuid
import warnings
from itertools import chain, islice, repeat

import torch

# # How to generate these lists:
# from itertools import islice
# from matsign.methods import OursFixedL, Ours
# hs = list(OursFixedL(l=1e-3, cushion=1e-1, center_squred_svs=False, max_iters=10)(1e-3))  # centered
# hs = list(islice(Ours(cushion=1e-1, center_squred_svs=False).uncentered_sequence(1e-3), 10))  # uncentered
# [tuple(float(x) for x in h.coef) for h in hs]

# 17, 17, 17
co17 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
]

# 17, 17, 9
co9 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [[8.18628571, -7.34995052], [0.08376457, -4.48494194, -4.06476615]],
        [[1.33726249, -0.96757271], [-6.2348802, 0.38265358, 0.0163027]],
        [-1.48495011, -0.04376982, 1.0, -1.0],
    ),
]

# 17,17,5
co5 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (2.64972986, -1.93611987, 0.43470742),
]

# safety factor for numerical stability (but exclude last polynomial)

for i in range(len(co5) - 1):
    for j in range(len(co5[i][2])):
        co5[i][2][j] /= 1.01 ** (j + 2)
for i in range(len(co9) - 1):
    for j in range(len(co9[i][2])):
        co9[i][2][j] /= 1.01 ** (j + 2)
for i in range(len(co17) - 1):
    for j in range(len(co17[i][2])):
        co17[i][2][j] /= 1.01 ** (j + 2)


"""coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]
] + [coeffs_list[-1]]"""


@torch.compile
def MachPolar(G: torch.Tensor, steps: int) -> torch.Tensor:
    co = co5
    m_list = [3,3,1]
    if steps == 17:
        co = co17
        m_list = [3,3,3]
    elif steps == 9:
        co = co9
        m_list = [3,3,2]
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    n = X.shape[0]
    t = 0

    for m in m_list:
        if m == 1:
            A = X @ X.mT
            B = co[2][1] * A + co[2][2] * A @ A
            X = co[2][0] * X + B @ X
            continue

        A = co[t][0]
        B = co[t][1]
        c = co[t][2]
        t += 1
        out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)
        out[0] = torch.eye(n, dtype=X.dtype, device=X.device)  # "1" as identity matrix
        out[1] = X @ X.mT
        for i in range(m):
            out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 @ out2)

        X = torch.sum(out, dim=0) @ X
    """if steps == 3:
        A = X @ X.mT
        B = co[2][1] * A + co[2][2] * A @ A
        X = co[2][0] * X + B @ X"""
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


"""
@torch.compile
def FastApplyPolarExpress(
    G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0
) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.double()
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    hs = [(a * 0.99, b * 0.99, c * 0.99) for (a, b, c) in hs]  # safety factor
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    Y = X @ X.mT + shift_eps * I  # numerical stability
    Q = I.clone()
    for iter, (a, b, c) in enumerate(hs):
        if (iter % restart_interval == 0) and (iter > 0):
            X = Q @ X
            Y = X @ X.mT
            Q = I.clone()
        R = Q.mT @ Y @ Q
        Q = Q @ (a * I + R @ (b * I + c * R))  # Q <- Q(aI + bR + cR^2)
        # if verbose:
        #     print("-"*20)
        #     print(iter)
        #     print("R", torch.linalg.eigvalsh(R.double())[:10])
        #     print((R - R.T).norm().item())
        #     print("Q", torch.linalg.eigvalsh(Q.double())[:10])
        #     print((Q - Q.T).norm().item())
        #     print(torch.linalg.norm((Q @ X).double(), ord=2).item())
    X = Q @ X
    if (X.norm(dim=(-2, -1), keepdim=False) > 5 * I.shape[0]).any() or not (
        torch.isfinite(X).all()
    ):
        warnings.warn("X.norm() is unusually large. Saving G to disk.")
        os.makedirs("bad_G", exist_ok=True)
        filename = f"bad_G_{uuid.uuid4().hex}.pt"
        torch.save(G, os.path.join("bad_G", filename))
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
"""
