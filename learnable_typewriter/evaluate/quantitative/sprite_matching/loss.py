# Needs cleaning possibly file structure?
import torch

def matching_loss_batch(x, y, x_length, y_length, C, skip):
    B, N, M = x.size()[-1], x.size()[0], y.size()[0]
    D = torch.zeros((N+1, M+1, B)).to(x.device)
    D[D == 0] = float('inf')
    for i in range(N+1):
        for j in range(M+1):
            if i == 0 and j == 0:
                D[0, 0] = 0
            elif i == 0:
                D[0, j] = j*skip[1]
            elif j == 0:
                D[i, 0] = i*skip[0]
            else:
                i1, i2 = x[i-1], y[j-1]
                D[i, j] = torch.min(torch.cat([
                    (C[i1, i2] + D[i-1, j-1]).unsqueeze(0),
                    (D[i-1, j] + skip[0]).unsqueeze(0),
                    (D[i, j-1] + skip[1]).unsqueeze(0)], dim=0),
                    dim=0)[0]

    return D[x_length+1, y_length+1, torch.arange(B)].sum()

def matching_loss(x, y, x_length, y_length, C, skip):
    B = x_length.size()[0]
    S = 0
    for b in range(B):
        x1, x2 = x[:x_length[b],b].tolist(), y[:y_length[b],b].tolist()
        N, M = x_length[b], y_length[b]
        D = torch.zeros((N+1, M+1))
        D[D == 0] = float('inf')
        for i in range(N+1):
            for j in range(M+1):
                if i == 0 and j == 0:
                    D[0, 0] = 0
                elif i == 0:
                    D[0, j] = j*skip[1]
                elif j == 0:
                    D[i, 0] = i*skip[0]
                else:
                    i1, i2 = x1[i-1], x2[j-1]
                    D[i, j] = min(C[i1, i2] + D[i-1, j-1], D[i-1, j] + skip[0], D[i, j-1] + skip[1])

        S += D[-1, -1]
    return S