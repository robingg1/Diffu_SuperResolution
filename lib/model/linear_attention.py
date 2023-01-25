import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

    
# embedding time steps
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# encoder part to encode B*3*num image feat to B*1*num feature map
class PointEncoder(nn.Module):
    def __init__(
        self,
        input_channels = 272,
        hiddem_channels = 128,
        layers = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hiddem_channels
        self.encoder_input = nn.Sequential(
            nn.Conv1d(self.input_channels,hiddem_channels,3,1,1),
            nn.SiLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv1d(self.hidden_channels,self.hidden_channels,3,1,1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels,int(self.hidden_channels/2),3,1,1),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Conv1d(int(self.hidden_channels/2),3,3,1,1),
            nn.SiLU(),
            nn.Conv1d(3,1,3,1,1),
        )

    def forward(self,x, residual=False):
        x = self.encoder_input(x)
        x = self.middle(x)
        if residual:
            middle = x
            x = self.output(x)
            x = x+middle
        else:
            x = self.output(x)
        #x = torch.squeeze(x,1)

        return x

class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=8000,
                 text_latent_dim=8000,
                 time_embed_dim=2000,
                 ffn_dim=1000,
                 num_head=2,
                 dropout=0.3):
        super().__init__()
        # self.sa_block = LinearTemporalSelfAttention(
        #     seq_len, latent_dim, 8, 0.1, time_embed_dim)
        self.dropout = 0.3
        self.num_head = num_head
        self.ca_block = LinearTemporalCrossAttention(
            latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb):
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x


        


class OneDUnet(nn.Module):
    def __init__(
        self,
        xt_dim = 8000,
        time_dim = None,
        cond_dim = 8000,
        num_layers = 1,
        max_steps = 200
        
    ):
        super().__init__()

        self.pointencoder = PointEncoder()
        self.xt_dim = xt_dim
        self.cond_dim = cond_dim
        if time_dim == None:
            self.time_dim = int(self.xt_dim/4)
        else:
            self.time_dim = time_dim
        self.condencoder = nn.Sequential(
            nn.Linear(self.cond_dim,self.time_dim),
            nn.Linear(self.time_dim,self.time_dim)
        )
        self.max_steps = max_steps
        self.text_ln = nn.LayerNorm(cond_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(cond_dim, self.time_dim)
        )

        # Input Embedding
        self.joint_embed = nn.Linear(self.xt_dim, self.xt_dim)

        self.attention_decoder = LinearTemporalDiffusionTransformerDecoderLayer()
        self.time_embed = nn.Sequential(
            nn.Linear(self.xt_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.temporal_decoder_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.temporal_decoder_blocks.append(self.attention_decoder)


        self.concat_decoder = nn.Sequential(
            nn.Linear(self.xt_dim+self.time_dim+self.cond_dim,int((self.xt_dim+self.time_dim+self.cond_dim)/2)),
            nn.SiLU(),
            nn.LayerNorm(int((self.xt_dim+self.time_dim+self.cond_dim)/2)),
            nn.Linear(int((self.xt_dim+self.time_dim+self.cond_dim)/2),self.xt_dim),
            nn.SiLU(),
            nn.LayerNorm(self.xt_dim),
            nn.Linear(self.xt_dim,self.xt_dim),
            nn.SiLU(),
            nn.LayerNorm(self.xt_dim),
        )

        self.out = nn.Sequential(nn.LayerNorm(self.xt_dim),(nn.Linear(self.xt_dim, self.xt_dim)),nn.Sigmoid()
        )

    def forward(self, x, time_steps, cond, use_attention=True):
        time_embed = self.time_embed(timestep_embedding(time_steps,self.xt_dim,self.max_steps))
        x_embed = self.joint_embed(x)
        cond_embed = self.pointencoder(cond)
        # print(x_embed)
        # print('!!!!!!')
        # print(cond_embed)
        if use_attention:
          for module in self.temporal_decoder_blocks:
             x_embed = module(x_embed,cond_embed,time_embed)
             
        else:
            time_embed = torch.unsqueeze(time_embed,1)
            concat_x = torch.concat((x_embed,cond_embed,time_embed),2)
            x_embed = self.concat_decoder(concat_x)

        output = self.out(x_embed)
        return output
