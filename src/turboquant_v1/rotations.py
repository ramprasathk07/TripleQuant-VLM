import torch 
from torch import Tensor as T

def fwht(x: T, normalize: bool = True) -> T:

    d = x.shape[-1]

    if (d & (d - 1)) != 0:
        raise ValueError("dimension must be power of 2")

    y = x.clone()

    h = 1

    while h < d:

        y = y.reshape(*y.shape[:-1], -1, h * 2)

        a = y[..., :, :h].clone()
        b = y[..., :, h:2*h].clone()

        y[..., :, :h] = a + b
        y[..., :, h:2*h] = a - b

        y = y.reshape(*x.shape)

        h *= 2

    if normalize:
        y = y / (d ** 0.5)

    return y


def generate_rotation_hadamard_matrix(
    d: int,
    device=None,
    dtype=torch.float32,
    seed: int = 42
) -> T:
    """
    Generate explicit randomized Hadamard rotation matrix:

        R = H D

    where:
        H = normalized Hadamard
        D = random sign diagonal matrix
    """

    if (d & (d - 1)) != 0:
        raise ValueError("dimension must be power of 2")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # random diagonal signs
    signs = torch.randint(
        0,
        2,
        (d,),
        generator=g
    ).to(dtype)

    signs = signs * 2 - 1

    # identity matrix
    I = torch.eye(d, dtype=dtype)

    # apply D
    I = I * signs

    # apply H
    R = fwht(I)

    if device is not None:
        R = R.to(device)

    return R

def generate_rotation_matrix(
    d: int,
    device : torch.device,
    dtype : torch.dtype = torch.float32,
    seed : int = 42   
)-> T:
    
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    G = torch.randn(
        d, 
        d,
        generator=rng,
        dtype=torch.float32
    )

    Q,R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)

def generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    """
    Generate the random projection matrix S ∈ R^{d×d} for QJL.
    S has i.i.d. N(0,1) entries.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply random rotation: y = x @ Pi^T (equivalent to Pi @ x for each vector)."""
    return torch.matmul(x, Pi.T)


def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation: x = y @ Pi (equivalent to Pi^T @ y)."""
    return torch.matmul(y, Pi)


if __name__ == "__main__":
    # x = torch.randn(2, 4096, device="cuda")
    # y = randomized_hadamard_rotation(x)
    # print(y.shape)
    y = generate_rotation_hadamard_matrix(4)
    z = generate_rotation_matrix(4, "cuda")
    print(y.shape)
    print(z.shape)
