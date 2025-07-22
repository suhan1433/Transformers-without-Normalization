import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        # q = (n_splits x bs, m, hs//n_split)
        # mask = (n_splits x bs, m, n)

        w = torch.bmm(Q, K.transpose(-2,-1)) # (bs, m, hs) (bs, hs, n) -> (bs, m(디코더 timestep), n(인코더))
        
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))
        
        w = self.softmax(w / (dk**.5)) # (bs, m, n)
        c = torch.bmm(w,V) # (bs x n_splits, m, hs//n_splits)

        return c



class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super(MultiHead, self).__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        self.Q_linear = nn.Linear(hidden_size,hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size,hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size,hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        # (batch_size, m, hidden_size//n_split)
        # QWs는 리스트가 된다.
        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)

        QWs= torch.cat(QWs, dim=0)
        KWs= torch.cat(KWs, dim=0)
        VWs= torch.cat(VWs, dim=0)

        if mask is not None:
            # mask = (bs* n_split, m, n)
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # 인코더는 pad부분을 디코더도 동일하게 pad인 부분 
        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk= self.hidden_size // self.n_splits,
        )

        # multi-header concat
        c = c.split(Q.size(0), dim =0)
        # c_i = (bs, m, hidden/n_splits)
        c = self.linear(torch.cat(c, dim=-1))
        # c = (bs, m, hidden)
        return c
