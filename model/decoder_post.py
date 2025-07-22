import torch
import torch.nn as nn
from model.multihead_attention import MultiHead

class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p = .1,
        use_leaky_relu=False,
    ):
        super(DecoderBlock, self).__init__()

        # self-attention
        self.masked_attn = MultiHead(hidden_size, n_splits)
        self.masked_attn_norm = nn.LayerNorm(hidden_size)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        # cross attention
        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu == True else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, k_and_v, mask, prev, future_mask):
        # k_and_value = (bs, n, hidden) 인코더 결과
        # mask = (bs, m, n) 인코더에서의 빈 타임스템 pad를 마스킹 해놓은 마스크

        if prev is None: # training
            # x             = (bs, m, hiddensize)
            # prev          = None
            # future_mask   = (batch_size, m, m)
            # z             = (bs, m, hidden)

            # post-LN
            z = self.masked_attn_norm(x + self.masked_attn_dropout(
                 self.masked_attn(x, x, x, mask=future_mask)
                 ))

        else : # inference
            # 한번 할 떄 하나의 단어를 보고 다음 단어를 에측하기에(auto regressive) seq_len = 1
            # x             = (bs, 1, hiddensize)
            # prev          = (bs, t-1, hidden_size)
            # future_mask   =  None
            # z             = (bs, 1, hidden)
            # post-LN
            z = self.masked_attn_norm(x + self.masked_attn_dropout(self.attn(x, prev, prev, mask=None)))
        
        # post-LN
        # 인코더의 pad부분에 채워준다
        z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z,K = k_and_v, V = k_and_v, mask=mask)))
        ## z = (bs, hs)

        ## post-LN
        z = self.fc_norm(z + self.fc_dropout(self.fc(z)))

        # 다음 레이어에서 사용할 수 있도록 똑같은 아웃풋
        return z, k_and_v, mask, prev, future_mask

        