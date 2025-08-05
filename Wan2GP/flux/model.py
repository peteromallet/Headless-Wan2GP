from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def preprocess_loras(self, model_type, sd):
        new_sd = {}
        if len(sd) == 0: return sd

        def swap_scale_shift(weight):
            shift, scale = weight.chunk(2, dim=0)
            new_weight = torch.cat([scale, shift], dim=0)
            return new_weight

        first_key= next(iter(sd))
        if first_key.startswith("lora_unet_"):
            new_sd = {}
            print("Converting Lora Safetensors format to Lora Diffusers format")
            repl_list = ["linear1", "linear2", "modulation", "img_attn", "txt_attn", "img_mlp", "txt_mlp", "img_mod", "txt_mod"]
            src_list = ["_" + k + "." for k in repl_list]
            src_list2 = ["_" + k + "_" for k in repl_list]
            tgt_list = ["." + k + "." for k in repl_list]

            for k,v in sd.items():
                k = k.replace("lora_unet_blocks_","diffusion_model.blocks.")
                k = k.replace("lora_unet__blocks_","diffusion_model.blocks.")
                k = k.replace("lora_unet_single_blocks_","diffusion_model.single_blocks.")
                k = k.replace("lora_unet_double_blocks_","diffusion_model.double_blocks.")

                for s,s2, t in zip(src_list, src_list2, tgt_list):
                    k = k.replace(s,t)
                    k = k.replace(s2,t)

                k = k.replace("lora_up","lora_B")
                k = k.replace("lora_down","lora_A")

                new_sd[k] = v

        elif first_key.startswith("transformer."):
            root_src = ["time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2", "time_text_embed.text_embedder.linear_1", "time_text_embed.text_embedder.linear_2",
                    "time_text_embed.guidance_embedder.linear_1", "time_text_embed.guidance_embedder.linear_2",
                    "x_embedder", "context_embedder", "proj_out" ]

            root_tgt = ["time_in.in_layer", "time_in.out_layer", "vector_in.in_layer", "vector_in.out_layer",
                    "guidance_in.in_layer", "guidance_in.out_layer",
                    "img_in", "txt_in", "final_layer.linear" ]

            double_src = ["norm1.linear", "norm1_context.linear", "attn.norm_q",  "attn.norm_k", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2", "attn.to_out.0" ,"attn.to_add_out", "attn.to_out", ".attn.to_", ".attn.add_q_proj.", ".attn.add_k_proj.", ".attn.add_v_proj.",  ] 
            double_tgt = ["img_mod.lin", "txt_mod.lin", "img_attn.norm.query_norm", "img_attn.norm.key_norm", "img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2", "img_attn.proj", "txt_attn.proj", "img_attn.proj", ".img_attn.", ".txt_attn.q.", ".txt_attn.k.", ".txt_attn.v."] 

            single_src = ["norm.linear", "attn.norm_q", "attn.norm_k", "proj_out",".attn.to_q.", ".attn.to_k.", ".attn.to_v.", ".proj_mlp."]
            single_tgt = ["modulation.lin","norm.query_norm", "norm.key_norm", "linear2", ".linear1_attn_q.", ".linear1_attn_k.", ".linear1_attn_v.", ".linear1_mlp."]


            for k,v in sd.items():
                if k.startswith("transformer.single_transformer_blocks"):
                    k = k.replace("transformer.single_transformer_blocks", "diffusion_model.single_blocks")
                    for src, tgt in zip(single_src, single_tgt):
                        k = k.replace(src, tgt)
                elif k.startswith("transformer.transformer_blocks"):
                    k = k.replace("transformer.transformer_blocks", "diffusion_model.double_blocks")
                    for src, tgt in zip(double_src, double_tgt):
                        k = k.replace(src, tgt)
                else:
                    k = k.replace("transformer.", "diffusion_model.")
                    for src, tgt in zip(root_src, root_tgt):
                        k = k.replace(src, tgt)

                    if "norm_out.linear" in k:
                        if "lora_B" in k:
                            v = swap_scale_shift(v)
                        k = k.replace("norm_out.linear", "final_layer.adaLN_modulation.1")            
                new_sd[k] = v
        return new_sd    

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        callback= None,
        pipeline =None,

    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec +=  self.guidance_in(timestep_embedding(guidance, 256))
        vec +=  self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            if callback != None:
                callback(-1, None, False, True)
            if pipeline._interrupt:
                return None
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
