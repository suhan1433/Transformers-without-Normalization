from model.multihead_attention import MultiHead
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p = .1,
        use_leaky_relu=False,
        ):
        super(EncoderBlock, self).__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        # feedforward
        self.fc = nn.Sequential(
            # 512 * 4 = 2048
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)
    
    def forward(self, x, mask):
        # x = (bs, n, hidden)
        # mask  = (bs, n, n)

        # post-LN
        # layernorm(add + multi)
        # z = self.attn_norm(x + self.attn_dropout(self.attn(Q=x, K=x, V=x, mask = mask)))
        # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
        
        # Pre-LN
        z = self.attn_norm(x)
        z = x + self.attn_dropout(self.attn(Q=z, K=z, V=z, mask = mask))
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))

        # z = (bs, n , hiddenz-size)

        return z, mask




         
