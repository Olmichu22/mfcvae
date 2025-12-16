
import torch
import torch.nn as nn
from .utils import build_fc_network
from typing import List
from torch_geometric.nn import global_mean_pool, global_max_pool  # o global_max_pool
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from gatr import GATr, SelfAttentionConfig, MLPConfig
from xformers.ops.fmha import BlockDiagonalMask
import torch_scatter

class ActivationWithBatch(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.act = activation
    def forward(self, x, batch=None):
        return self.act(x)

class IdentityWithBatch(nn.Module):
    def forward(self, x, batch=None):
        return x
    
class SequentialWithBatch(nn.Sequential):
    def forward(self, x, batch=None):
        for module in self:
            try:
                x = module(x, batch)
            except TypeError:
                # el módulo no acepta batch → usar solo x
                x = module(x)
        return x

class AttentionPool(nn.Module):
    """Global attention pooling: softmax(aᵀ linear(x))"""
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Linear(in_dim, 1)

    def forward(self, x, batch):
        # scores por nodo
        w = self.att(x).squeeze(-1)        # (N,)

        # softmax por grafo usando scatter_softmax
        w = torch_scatter.scatter_softmax(w, batch)  # (N,)

        # sumar ponderado por grafo
        g = torch_scatter.scatter_sum(x * w.unsqueeze(-1), batch, dim=0)

        return g


class GlobalAggregator(nn.Module):
    """
    Global summary + aggregation back to each node.

    summary_method ∈ {"mean", "max", "attn"}
    agg_method     ∈ {"project", "concat", "none"}
    """
    def __init__(self, in_dim, summary_method="mean", agg_method="project"):
        super().__init__()

        # ---------- resumen global ----------
        summary_ops = {
            "mean": lambda x, b: global_mean_pool(x, b),
            "max":  lambda x, b: global_max_pool(x, b),
            "attn": AttentionPool(in_dim)
        }
        if summary_method not in summary_ops:
            raise ValueError("summary_method must be mean/max/attn")
        self.summary = summary_ops[summary_method]

        # ---------- agregación ----------
        if agg_method == "project":
            self.op = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim)
            )
            self.combine = lambda x, g: x + self.op(g)

        elif agg_method == "concat":
            self.op = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim)
            )
            self.combine = lambda x, g: self.op(torch.cat([x, g], dim=-1))

        elif agg_method == "none":
            self.op = IdentityWithBatch()
            self.combine = lambda x, g: x  # identidad

        else:
            raise ValueError("agg_method must be project/concat/none")

    def forward(self, x, batch):
        # resumen global por evento (B, F)
        if isinstance(self.summary, nn.Module):
            g = self.summary(x, batch)
        else:
            g = self.summary(x, batch)

        # expandimos al tamaño del batch (N, F)
        g_expanded = g[batch]

        return self.combine(x, g_expanded)


class GATrModule(nn.Module):
    def __init__(self,
                 in_mv_channels=1,
                 out_mv_channels=1,
                 hidden_mv_channels=32,
                 in_s_channels=2,
                 out_s_channels=16,
                 hidden_s_channels=64,
                 num_blocks=1,
                 attention: SelfAttentionConfig = SelfAttentionConfig(),
                 mlp: MLPConfig = MLPConfig(),
                 input_dim: int = 3,
                 emb_p = None,
                 do_bn = True,
                 dropout_prob = 0.1,
                 agg_config = {}):
        super().__init__()
        # print(num_blocks)
        # print(in_mv_channels,
        #       out_mv_channels,
        #       hidden_mv_channels,
        #         in_s_channels,
        #         out_s_channels,
        #         hidden_s_channels
        #       )
        self.gatr = GATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=attention,
            mlp=mlp,
            # checkpoint = ["block"]
        )

        # BatchNorm sobre coordenadas de entrada
        # print("CapaDropout?")
        self.dropout = nn.Dropout(dropout_prob)
        # print("CapaDropout?")
        
        self.emb_p = emb_p
        self.do_bn = do_bn
        self.agg_config = agg_config
        if self.do_bn:
            if self.emb_p == "dec":
                channels = 4 + out_s_channels
                self.pos_bn = nn.BatchNorm1d(channels, momentum=0.1)
            else:
                channels = 16 + out_s_channels
                
                self.pos_bn = nn.BatchNorm1d(channels, momentum=0.1)
        else:
            self.pos_bn = IdentityWithBatch()
        
        if self.emb_p:
            if self.emb_p == "enc":
                self.enc_x = self.encode_x_GA
                self.dec_x = self.concat_geom_vars
                
            elif self.emb_p == "dec":
                self.enc_x = self.extract_geom_vars
                self.dec_x = self.decode_x_GA
            else:
                raise ValueError("Emb_p option is not valid for GATr Module")
        else:
            self.enc_x = self.extract_geom_vars
            self.dec_x = self.concat_geom_vars
        
        self.output_dim = 0
        if self.emb_p == "dec":
            self.output_dim = 4 + out_s_channels
        else:
            self.output_dim = 16 + out_s_channels
        
        if self.emb_p == "dec":
            agg_method = "none"
        else:
            agg_method = agg_config.get("agg", "none")    
        self.agg = GlobalAggregator(in_dim=self.output_dim,
                                    summary_method=agg_config.get("summary", "mean"),
                                    agg_method=agg_method)
    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"
    
    def extra_repr(self):
        info = self.emb_p if self.emb_p is not None else "none"
        summary = self.agg_config.get('summary', 'mean')
        agg = self.agg_config.get('agg', 'none')
        return f"GA Mode={info}, summary={summary}, agg={agg}"
            
    def extract_geom_vars(self, input):
        embedded_mv = input[:,:16]
        embedded_scalars = input[:,16:]
        embedded_mv = embedded_mv.unsqueeze(-2)
        return embedded_mv, embedded_scalars
    
    def concat_geom_vars(self, embedded_mv, embedded_scalars):
        x_latent = torch.cat([embedded_mv[:,0,:], embedded_scalars], dim=-1) # (N, 32)
        return x_latent
    
    def encode_x_GA(self, input):
        embedded_points = embed_point(input[:,:3])
        embedded_scalar = embed_scalar(input[:,3:4])
        extra_scalars = input[:,4:]
        
        embedded_geom = embedded_points + embedded_scalar
        embedded_geom = embedded_geom.unsqueeze(-2)

        return embedded_geom, extra_scalars
        
    def decode_x_GA(self, embedded_mv, embedded_scalars):
        points = extract_point(embedded_mv[:, 0, :]) # (N, 3)
        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_mv).squeeze(dim=2)  # (..., num_points, 1, 1) (N, 1)
        x_point = points
        x_scalar = torch.cat(
            (nodewise_outputs, embedded_scalars), dim=-1 
        )
        x_dec = torch.cat([x_point, x_scalar], dim=-1) # dim (N, 3 + 1 + s_out == dim out)
        return x_dec
        
    def build_attention_mask(self, batch):
        batch_numbers = batch
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )
    def make_momentum_0(self):
        if self.do_bn:
            self.pos_bn.momentum=0
        
    def forward(self, input_vars, batch):
        """
        If input_vars is in Latent Space, input_vars[:,:16] is the mv and the rest the scalars.
        If input_vars is in real space, input_vars[:, :3] is the point, and the rest the scalars (first scalar is geom).
        pos:         (N_total, pos_dim)
        feats:       (N_total,)         # escalar por nodo
        extra_feats: (N_total, k_extra)
        batch:       (N_total,)         # índices de grafo

        returns:
            x_latent: (N_total, F_latent)
        """
        # 1) normalizar coords
        
        # inputs = self.pos_bn(input_vars)                    # (N, pos_dim)
        inputs =  input_vars
        # inputs = pos
        # 2) escalar "feats" → (N,1)
        # feats = feats.view(-1, 1)         # (N,1)
        # extra_feats = extra_feats.view(-1, 1)
        
        # 3) embedding geométrico
        embedded_geom, scalars = self.enc_x(inputs)

        # self attention mask
        mask = self.build_attention_mask(batch)

        # # 6) pasada GATr
        embedded_mv, embedded_scalars = self.gatr( # (N, 16) (N, 16)
            embedded_geom, scalars=scalars, attention_mask=mask
        )
        x_latent = self.dec_x(embedded_mv, embedded_scalars)
        x_latent = self.agg(x_latent, batch)
        x_latent = self.pos_bn(x_latent)                    # (N, pos_dim)
        x_latent = self.dropout(x_latent)
        return x_latent
    
def build_gatr_network(layer_dims, activation, dropout_prob, batch_norm, gatr_config, emb_p="enc"):
    net = []
    for i in range(1, len(layer_dims)):
        if i == 1 and emb_p=="enc":
            gatr_config["emb_p"] = "enc"
            gatr_config["in_s_channels"] = layer_dims[i-1] - 4 # total dims - 3 for the coords and -1 for the scalar geom feature
            gatr_config["out_s_channels"] = layer_dims[i] - 16 # total dims -16 for mv embedding

        elif i == (len(layer_dims)-1) and emb_p=="dec":
            gatr_config["emb_p"] = "dec"
            gatr_config["in_s_channels"] = layer_dims[i-1] - 16 # total dims -16 for mv embedding
            gatr_config["out_s_channels"] = layer_dims[i] - 4 # total dims - 3 for the extract_point and scalar features

        else:
            gatr_config["emb_p"] = None
            gatr_config["in_s_channels"] = layer_dims[i-1] - 16 # total dims -16 for mv embedding
            gatr_config["out_s_channels"] = layer_dims[i] - 16 # total dims -16 for mv embedding
            
            
        gatr_config["do_bn"] = batch_norm
        gatr_config["dropout_prob"] = dropout_prob
            
        gatr = GATrModule(**gatr_config)
        net.append(gatr)
        if activation == "relu":
            net.append(ActivationWithBatch(nn.ReLU()))
        elif activation == 'leaky_relu':
            net.append(ActivationWithBatch(nn.LeakyReLU()))
            
        elif activation == 'elu':
            net.append(ActivationWithBatch(nn.ELU()))
            
        elif activation == "sigmoid":
            net.append(ActivationWithBatch(nn.Sigmoid()))
            
        elif activation == "gelu":
            net.append(ActivationWithBatch(nn.GELU()))
            
    net = SequentialWithBatch(*net)
    return net
# def build_gatr_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm, emb_p = None):
    # emb_p puede ser enc o dec para que el modelo transforme de nuevo al espacio real o el geométrico
    

class GATrEncoder(nn.Module):
    """
    Encoder geométrico + pasada de GATr.
    Produce x_latent (embedding fijo de entrada para TRM).
    """
    def __init__(self, 
                 layer_dims,
                 in_dim,
                 activation,
                 do_fc_batch_norm,
                 dropout_prob = 0.,
                 gatr_config={}):
        
        super(self.__class__, self).__init__()
        
        self.J_n_mixtures = len(layer_dims)
        self.fc_backbone = nn.ModuleList()  # "enc"
        self.fc_rung = nn.ModuleList()  # "qladder" / "Sprosse"
        self.encoder_output_dims = []
        self.gatr_config = gatr_config
        
        
        # construct network
        b_lower_dim = in_dim
        self.rung_adapter = nn.ModuleList()
        for j in range(self.J_n_mixtures):
            branch_index = layer_dims[j].index("branch")  # find branch
            # print(b_lower_dim)
            backbone_dims = [b_lower_dim] + layer_dims[j][:branch_index]
            # update b_lower_dim here already
            b_lower_dim = layer_dims[j][branch_index - 1] if branch_index-1 >= 0 else b_lower_dim  # lower branch index; used in upper layer
            rung_dims = [b_lower_dim] + layer_dims[j][branch_index + 1:]

            # print(backbone_dims)
            # print(rung_dims)

            if len(backbone_dims) == 1:  # only input dimension
                self.fc_backbone.append(IdentityWithBatch())
            else:
                self.fc_backbone.append(build_gatr_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm, gatr_config=self.gatr_config, emb_p = "enc"))

            if len(rung_dims) == 1:  # only input dimension
                self.fc_rung.append(IdentityWithBatch())
            else:
                self.fc_rung.append(build_gatr_network(layer_dims=rung_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm, gatr_config=self.gatr_config))

            # AÑADIDO EL RUNG ADAPTER

            adapter_dim = rung_dims[-1]  # puedes dejarlo igual o reducir
            self.rung_adapter.append(
                nn.Sequential(
                    nn.Linear(rung_dims[-1], adapter_dim),
                    # nn.GELU(),
                    # nn.Linear(adapter_dim, adapter_dim)
                )
            )
            self.encoder_output_dims.append(rung_dims[-1])


    # AÑADIDO EL RUNG ADAPTER
    def forward(self, x, batch):
        # print("forward start ---")
        rung_list = []
        b = x
        for j in range(self.J_n_mixtures):
            # print(b.size())
            b = self.fc_backbone[j](b, batch)
            if self.do_progressive_training:
                b_aux = b * self.alpha_enc_fade_in_list[j]
            else:
                b_aux = b
            r = self.fc_rung[j](b_aux, batch)
            r = self.rung_adapter[j](r) 
            rung_list.append(r)

        return rung_list

class GATrDecoder(nn.Module):
    def __init__(self, layer_dims: List[int], z_j_dim_list: List[int], merge_type: str = 'gated_add',
                 activation: str = "relu", dropout_prob: float = 0., do_fc_batch_norm: bool = False, gatr_config: dict = {}):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = len(layer_dims)
        self.z_dim_list = z_j_dim_list
        self.merge_type = merge_type
        self.fc_backbone = nn.ModuleList()  # "dec"
        self.fc_rung = nn.ModuleList()  # "pladder" / "Sprosse"
        self.gatr_config = gatr_config
        # construct network
        for j in range(self.J_n_mixtures):
            if j == self.J_n_mixtures - 1:
                # edge case: no 'merge' here
                # whether it's rung or backbone is arbitrary here
                rung_dims = []
                backbone_dims = [self.z_dim_list[j]] + layer_dims[j]
            else:
                merge_index = layer_dims[j].index("merge")  # find branch
                # print(merge_index)
                # note the reversed order!
                rung_dims = [self.z_dim_list[j]] + layer_dims[j][:merge_index]
                if self.merge_type == 'gated_add':
                    backbone_dims = [layer_dims[j][merge_index - 1]] + layer_dims[j][merge_index + 1:]
                elif self.merge_type == 'cat':
                    backbone_dims = [layer_dims[j][merge_index - 1] + layer_dims[j + 1][-1]] + layer_dims[j][merge_index + 1:]
            # print(backbone_dims)
            # print(rung_dims)
            if len(rung_dims) == 1:  # only input dimension
                self.fc_rung.append(IdentityWithBatch())
            else:
                self.fc_rung.append(build_gatr_network(layer_dims=rung_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm, gatr_config=self.gatr_config))

            if len(backbone_dims) == 1:  # only input dimension
                self.fc_backbone.append(IdentityWithBatch())
            else:
                self.fc_backbone.append(build_gatr_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm, gatr_config=self.gatr_config, emb_p="dec"))

    
    def merge(self, r, upper_b, merge_type='gated_add', const=0.1):
        if merge_type == 'gated_add':
            m = const * r + upper_b
        elif merge_type == 'cat':
            m = torch.cat((r, upper_b), dim=1)

        return m


    def forward(self, z_sample_q_z_j_x_list: List[torch.tensor], batch):
        b = z_sample_q_z_j_x_list[self.J_n_mixtures - 1]
        b = self.fc_backbone[self.J_n_mixtures - 1](b, batch)  # rung is empty here
        for j in reversed(range(self.J_n_mixtures - 1)):  # last one already processed
            r = self.fc_rung[j](z_sample_q_z_j_x_list[j], batch)
            if self.do_progressive_training:
                r_aux = r * self.alpha_dec_fade_in_list[j]
            else:
                r_aux = r
            b = self.merge(r_aux, b, merge_type=self.merge_type)
            b = self.fc_backbone[j](b, batch)

        return b
  



class FCsharedEncoder(nn.Module):
    def __init__(self, layer_dims: List[int], J_n_mixtures: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.net = build_fc_network(layer_dims=layer_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm)

    def forward(self, x):
        h = self.net(x)
        h_list = [h for _ in range(self.J_n_mixtures)]

        return h_list


class FCSharedDecoder(nn.Module):
    def __init__(self, layer_dims: List[int], J_n_mixtures: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.net = build_fc_network(layer_dims=layer_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm)

    def forward(self, z_sample_q_z_j_x_list: List[torch.tensor]):
        z_sample_q_z_x = torch.cat(z_sample_q_z_j_x_list, dim=1)

        return self.net(z_sample_q_z_x)


class FCseparateEncoders(nn.Module):
    def __init__(self, layer_dims: List[int], J_n_mixtures: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.net_list = nn.ModuleList()
        for j in range(self.J_n_mixtures):
            self.net_list.append(build_fc_network(layer_dims=layer_dims[j], activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))


    def forward(self, x):
        h_list = []
        for j in range(self.J_n_mixtures):
            h = self.net_list[j](x)
            h_list.append(h)

        return h_list


class FCvlaeEncoder(nn.Module):
    def __init__(self, layer_dims: List[int], in_dim: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = len(layer_dims)
        self.fc_backbone = nn.ModuleList()  # "enc"
        self.fc_rung = nn.ModuleList()  # "qladder" / "Sprosse"
        self.encoder_output_dims = []

        # construct network
        b_lower_dim = in_dim
        for j in range(self.J_n_mixtures):
            branch_index = layer_dims[j].index("branch")  # find branch
            # print(b_lower_dim)
            backbone_dims = [b_lower_dim] + layer_dims[j][:branch_index]
            # update b_lower_dim here already
            b_lower_dim = layer_dims[j][branch_index - 1] if branch_index-1 >= 0 else b_lower_dim  # lower branch index; used in upper layer
            rung_dims = [b_lower_dim] + layer_dims[j][branch_index + 1:]

            # print(backbone_dims)
            # print(rung_dims)

            if len(backbone_dims) == 1:  # only input dimension
                self.fc_backbone.append(nn.Identity())
            else:
                self.fc_backbone.append(build_fc_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

            if len(rung_dims) == 1:  # only input dimension
                self.fc_rung.append(nn.Identity())
            else:
                self.fc_rung.append(build_fc_network(layer_dims=rung_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

            self.encoder_output_dims.append(rung_dims[-1])



    def forward(self, x):
        # print("forward start ---")
        rung_list = []
        b = x
        for j in range(self.J_n_mixtures):
            # print(b.size())
            b = self.fc_backbone[j](b)
            if self.do_progressive_training:
                b_aux = b * self.alpha_enc_fade_in_list[j]
            else:
                b_aux = b
            r = self.fc_rung[j](b_aux)
            rung_list.append(r)

        return rung_list




class FCvlaeDecoder(nn.Module):
    def __init__(self, layer_dims: List[int], z_j_dim_list: List[int], merge_type: str = 'gated_add',
                 activation: str = "relu", dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = len(layer_dims)
        self.z_dim_list = z_j_dim_list
        self.merge_type = merge_type
        self.fc_backbone = nn.ModuleList()  # "dec"
        self.fc_rung = nn.ModuleList()  # "pladder" / "Sprosse"

        # construct network
        for j in range(self.J_n_mixtures):
            if j == self.J_n_mixtures - 1:
                # edge case: no 'merge' here
                # whether it's rung or backbone is arbitrary here
                rung_dims = []
                backbone_dims = [self.z_dim_list[j]] + layer_dims[j]
            else:
                merge_index = layer_dims[j].index("merge")  # find branch
                # print(merge_index)
                # note the reversed order!
                rung_dims = [self.z_dim_list[j]] + layer_dims[j][:merge_index]
                if self.merge_type == 'gated_add':
                    backbone_dims = [layer_dims[j][merge_index - 1]] + layer_dims[j][merge_index + 1:]
                elif self.merge_type == 'cat':
                    backbone_dims = [layer_dims[j][merge_index - 1] + layer_dims[j + 1][-1]] + layer_dims[j][merge_index + 1:]
            # print(backbone_dims)
            # print(rung_dims)
            if len(rung_dims) == 1:  # only input dimension
                self.fc_rung.append(nn.Identity())
            else:
                self.fc_rung.append(build_fc_network(layer_dims=rung_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

            if len(backbone_dims) == 1:  # only input dimension
                self.fc_backbone.append(nn.Identity())
            else:
                self.fc_backbone.append(build_fc_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

    def merge(self, r, upper_b, merge_type='gated_add', const=0.1):
        if merge_type == 'gated_add':
            m = const * r + upper_b
        elif merge_type == 'cat':
            m = torch.cat((r, upper_b), dim=1)

        return m


    def forward(self, z_sample_q_z_j_x_list: List[torch.tensor]):
        b = z_sample_q_z_j_x_list[self.J_n_mixtures - 1]
        b = self.fc_backbone[self.J_n_mixtures - 1](b)  # rung is empty here
        for j in reversed(range(self.J_n_mixtures - 1)):  # last one already processed
            r = self.fc_rung[j](z_sample_q_z_j_x_list[j])
            if self.do_progressive_training:
                r_aux = r * self.alpha_dec_fade_in_list[j]
            else:
                r_aux = r
            b = self.merge(r_aux, b, merge_type=self.merge_type)
            b = self.fc_backbone[j](b)

        return b



if __name__ == "__main__":

    enc = FCvlaeEncoder(layer_dims=[[100, 101, 'branch', 102, 103], [104, 105, 'branch'], ['branch'], ['branch', 106, 107]], in_dim = 50)
    x = torch.randn((10, 50))
    h_list = enc(x)

    dec = FCvlaeDecoder(layer_dims=[[100, 107, 'merge', 102, 103], [104, 107, 'merge'], [107, 'merge'], [109, 'merge', 106, 107], [108, 109]], z_j_dim_list= [3, 4, 5, 6, 7], merge_type='gated_add')
    z_sample_list = [torch.randn((10, 3)), torch.randn((10, 4)), torch.randn((10, 5)), torch.randn((10, 6)), torch.randn((10, 7))]
    h = dec(z_sample_list)
