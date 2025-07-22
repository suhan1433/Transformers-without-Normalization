from model.decoder import DecoderBlock
from model.encoder import EncoderBlock
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_splits,
        n_enc_blocks = 6,
        n_dec_blocks = 6,
        dropout_p = .1,
        use_leaky_relu = False,
        max_length= 512,
    ):
        super(Transformer, self).__init__()

        self.input_size = input_size #vocab size
        self.hidden_size = hidden_size 
        self.output_size = output_size #vocab size
        self.n_splits = n_splits
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_p = dropout_p
        self.max_length = max_length


        self.emb_enc = nn.Embedding(input_size, hidden_size)
        self.emb_dec = nn.Embedding(output_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_p)

        # 미니 배치마다 time step이 다르기에 그거에 맞게 잘라서 사용한다.
        self.pos_enc = self._generate_pos_enc(hidden_size, max_length)

        self.encoder = nn.ModuleList(
            [EncoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
            ) for _ in range(n_enc_blocks)]
        )


        self.decoder = nn.ModuleList(
            [DecoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
            ) for _ in range(n_enc_blocks)]
        )

      
        self.generator = nn.Sequential(
            #post_layer할때는 삭제할 것.
            nn.LayerNorm(hidden_size),  # pre_layerNorm Transformer에서만 softmax전 hidden layer에 norm을 해준다.
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    @torch.no_grad()
    def _generate_pos_enc(self, hidden_size, max_length):
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # enc = (max_length, hidden)

        pos = torch.arange(0, max_length).unsqueeze(-1).float() #마지막 차원 추가
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # pos = (max_length, 1)
        # dim = (1, hidden//2)

        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))

        return enc

    def _position_encoding(self, x, init_pos=0):
        # 학습 시에는 전체가 들어가기에 0부터 시작, 하지만 추론 시에는 하나가 들어가기에 0부터 시작하면 안됨
        # x = (bs, n , hidden)
        # self.pos_enc = (max_length(time step), hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc =self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
        # pos_enc = (1, n, hidden_size)
        
        # print("포즈인코딩 x", x.size())
        # print("pos_enc", pos_enc.size())
        # print("x.device", x.device)
        # print("pos device", pos_enc.to(x.device))
        # breakpoint()

        # x와 더해지면서 브로드캐스팅 되어서 1이 x.size(0)로 바뀜
        x = x + pos_enc.to(x.device)

        return x
    
    def forward(self, x, y, attention_mask_src):
        # x = (batch_size, n)
        # y = (batch_size, m)

        # 마스크가 pad부분에 어텐션 웨이트를 갖는 것을 막기 위해, 마스크는 그레디언트 계산 필요 x
        with torch.no_grad():
            # mask = self._generate_mask(x)
        
            # (Bs, n)
        
       
            # mask_enc = (bs,1 , n) -> (bs, n, n), 인코더에서 셀프어텐션
            # mask_dec = (bs, m, n) -> 디코더에서 인코더로 어텐션 시
            # mask_enc = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
            # mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))

            attention_mask_src = ~attention_mask_src.bool() # (bs, n)

            # mask_enc는 (batch_size, sequence_length, sequence_length) 형태로 변환
            mask_enc = attention_mask_src.unsqueeze(1).expand(*x.size(), attention_mask_src.size(-1))  # (batch_size, sequence_length, sequence_length)


            # mask_dec는 (batch_size, m, sequence_length) 형태로 변환
            mask_dec = attention_mask_src.unsqueeze(1).expand(*y.size(), attention_mask_src.size(-1))  # (batch_size, m, sequence_length)



        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        for encoder in self.encoder:
            z, _ = encoder(z, mask_enc)
            # z = (bs, n, hidden_size)
        
        # future mask
        with torch.no_grad():
            future_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()
            # future_mask = (m,m)
            future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
            # future_mask = (bs, m, m)
        
        h = self.emb_dropout(self._position_encoding(self.emb_dec(y)))
        for decoder in self.decoder:
            h, z, _, _, _ = decoder(h, z, mask_dec, None, future_mask)
        
        # h = (bs, m, output_size)

        y_hat = self.generator(h)
        # y_hat = (bs, m, output_size)
        return y_hat
