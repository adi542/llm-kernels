import torch

def rope(x,position,base=10000.0):
    N,D = x.shape
    half = D //2
    i = torch.arange(half,dtype=torch.float32, device=x.device)
    theta = base ** (-2 * i / D)
    angles = position[:,None].float() * theta[None,:]
    x1,x2 = x[:,:half],x[:,half:]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.cat([x1 * cos - x2 * sin,                    
                      x1 * sin + x2 * cos], dim=1)

def score(q, k, q_pos, k_pos):
    q_rot = rope(q, torch.tensor([q_pos]))
    k_rot = rope(k, torch.tensor([k_pos]))
    return (q_rot * k_rot).sum().item()

if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(1, 64)                                    # one random query
    k = torch.randn(1, 64)                                    # one random key

    print("Same distance (1 apart), different places:")
    for qp, kp in [(1, 0), (5, 4), (101, 100), (1000, 999)]:
        print(f"  query at {qp:4d}, key at {kp:4d}:  score = {score(q, k, qp, kp):.5f}")

    print("Different distances:")
    for qp, kp in [(1, 0), (2, 0), (10, 0), (100, 0)]:
        print(f"  query at {qp:4d}, key at {kp:4d}:  score = {score(q, k, qp, kp):.5f}")

    x = torch.randn(1, 64)
    print("Length before:", x.norm().item(), "  after:", rope(x, torch.tensor([37])).norm().item())
