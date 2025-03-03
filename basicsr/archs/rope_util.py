import torch

def init_t_xy(end_x: int, end_y: int):
    """
    Return two 1D arrays (t_x, t_y) of shape (N,)
    where N = end_x * end_y.
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """
    Returns two complex tensors: (freqs_cis_x, freqs_cis_y)
      - each is shape (N, dim//2) in complex form,
      - N = end_x * end_y,
      - 'dim' is the per-head dimension.

    We'll apply the first half of the channel to x-rotations,
    and the second half of the channel to y-rotations.
    """
    # half_dim is the "real dimension" for either the x or y axis
    half_dim = dim // 2

    # Each frequency array uses indices [0, 2, 4, ..., half_dim-2]
    # so that we have half_dim/2 complex pairs.
    freqs_x = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))

    # Make grid of positions (t_x, t_y)
    t_x, t_y = init_t_xy(end_x, end_y)  # each (N,)

    # Outer product => shape (N, half_dim/2) in float
    out_x = torch.outer(t_x, freqs_x)
    out_y = torch.outer(t_y, freqs_y)

    # Convert to complex via polar form: magnitude=1, angle=out_x or out_y
    freqs_cis_x = torch.polar(torch.ones_like(out_x), out_x)  # shape (N, half_dim//2) complex
    freqs_cis_y = torch.polar(torch.ones_like(out_y), out_y)  # shape (N, half_dim//2) complex

    return freqs_cis_x, freqs_cis_y

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    If x is shaped [B, H, N, D], then after we view x as complex
    it becomes [B, H, N, D/2] in complex. We want to broadcast
    'freqs_cis' which is [N, D/2] => [1,1,N,D/2].
    """
    # freq_cis: (N, d_complex), x: (B,H,N,d_complex)
    # We simply add leading dims of size 1
    return freqs_cis.unsqueeze(0).unsqueeze(0)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    x: (B, H, N, D_real), where D_real must be even so we can interpret
       half of that as the "complex dimension".
    freqs_cis: (N, D_real/2) in complex.

    We'll:
     1) reshape last dim to (..., D_real/2, 2) => interpret it as complex,
     2) broadcast-multiply,
     3) return real shape again.
    """
    # x.shape = [b, h, n, d], d must be even => d/2 complex channels
    # Convert from real to complex
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)  # => (b,h,n,d/2,2)
    x_c = torch.view_as_complex(x_)

    # Broadcast freq_cis => shape (1,1,n,d/2)
    freqs_cis = reshape_for_broadcast(freqs_cis, x_c)  # => (1,1,n,d/2) in complex

    # Multiply in complex
    x_out = x_c * freqs_cis  # => (b,h,n,d/2) in complex

    # Convert back to real 2D form => (b,h,n,d/2,2)
    x_out_real = torch.view_as_real(x_out)

    # Flatten last 2 dims back to size d
    x_out_real = x_out_real.reshape(*x_out_real.shape[:-2], -1)  # => (b,h,n,d)
    return x_out_real.type_as(x)
