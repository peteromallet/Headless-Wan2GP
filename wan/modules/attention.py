# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from importlib.metadata import version
from mmgp import offload
import torch.nn.functional as F


try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    flash_attn = None

try:
    from sageattention import sageattn_varlen
    def sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ):
        return sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    
except ImportError:
    sageattn_varlen_wrapper = None


import warnings

try:
    from sageattention import sageattn
    from .sage2_core import sageattn as alt_sageattn, is_sage2_supported
    sage2_supported =  is_sage2_supported()
except ImportError:
    sageattn = None
    alt_sageattn = None
    sage2_supported = False
# @torch.compiler.disable()
def sageattn_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    padding_length = q.shape[1] -attention_length
    q = q[:, :attention_length, :, : ] 
    k = k[:, :attention_length, :, : ]
    v = v[:, :attention_length, :, : ]
    if True:
        qkv_list = [q,k,v]
        del q, k ,v
        o = alt_sageattn(qkv_list, tensor_layout="NHD")
    else:
        o = sageattn(q, k, v, tensor_layout="NHD")
        del q, k ,v

    qkv_list.clear()

    if padding_length > 0:
        o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    return o

# try:
# if True:
    # from .sage2_core import sageattn_qk_int8_pv_fp8_window_cuda
    # @torch.compiler.disable()
    # def sageattn_window_wrapper(
    #         qkv_list,
    #         attention_length,
    #         window
    #     ):
    #     q,k, v = qkv_list
    #     padding_length = q.shape[0] -attention_length
    #     q = q[:attention_length, :, : ].unsqueeze(0)
    #     k = k[:attention_length, :, : ].unsqueeze(0)
    #     v = v[:attention_length, :, : ].unsqueeze(0)
    #     qkvl_list = [q, k , v]
    #     del q, k ,v
    #     o = sageattn_qk_int8_pv_fp8_window_cuda(qkvl_list, tensor_layout="NHD", window = window).squeeze(0)
    #     qkv_list.clear()

    #     if padding_length > 0:
    #         o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    #     return o
# except ImportError:
#     sageattn = sageattn_qk_int8_pv_fp8_window_cuda

@torch.compiler.disable()
def sdpa_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    padding_length = q.shape[1] -attention_length
    q = q[:attention_length, :].transpose(1,2)
    k = k[:attention_length, :].transpose(1,2)
    v = v[:attention_length, :].transpose(1,2)

    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=False
    ).transpose(1,2)
    del q, k ,v
    qkv_list.clear()

    if padding_length > 0:
        o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    return o


def get_attention_modes():
    ret = ["sdpa", "auto"]
    if flash_attn != None:
        ret.append("flash")
    if memory_efficient_attention != None:
        ret.append("xformers")
    if sageattn_varlen_wrapper != None:
        ret.append("sage")
    if sageattn != None and version("sageattention").startswith("2") :
        ret.append("sage2")
        
    return ret

def get_supported_attention_modes():
    ret = get_attention_modes()
    if not sage2_supported:
        if "sage2" in ret:
            ret.remove("sage2")
    return ret

__all__ = [
    'pay_attention',
    'attention',
]


def pay_attention(
    qkv_list,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    version=None,
    force_attention= None,
    cross_attn= False,
    k_lens = None
):

    attn = offload.shared_state["_attention"] if force_attention== None else force_attention
    q,k,v = qkv_list
    qkv_list.clear()

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if b > 0 and k_lens != None and attn in ("sage2", "sdpa"):
        # Poor's man var len attention
        chunk_sizes = []
        k_sizes = []
        current_size = k_lens[0]
        current_count= 1
        for k_len in k_lens[1:]:
            if k_len == current_size:
                current_count += 1
            else:
                chunk_sizes.append(current_count)
                k_sizes.append(current_size)
                current_count = 1
                current_size = k_len
        chunk_sizes.append(current_count)
        k_sizes.append(k_len)
        if len(chunk_sizes) > 1 or k_lens[0] != k.shape[1]:
            q_chunks =torch.split(q, chunk_sizes)
            k_chunks =torch.split(k, chunk_sizes)
            v_chunks =torch.split(v, chunk_sizes)
            q, k, v = None, None, None
            k_chunks = [ u[:, :sz] for u, sz in zip(k_chunks, k_sizes)]
            v_chunks = [ u[:, :sz] for u, sz in zip(v_chunks, k_sizes)]
            o = []
            for sub_q, sub_k, sub_v in zip(q_chunks, k_chunks, v_chunks): 
                qkv_list = [sub_q, sub_k, sub_v]
                sub_q, sub_k, sub_v = None, None, None
                o.append( pay_attention(qkv_list) )
            q_chunks, k_chunks, v_chunks = None, None, None
            o = torch.cat(o, dim = 0)
            return o
    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if attn=="sage" or attn=="flash":
        if b != 1 :
            if k_lens == None:
                k_lens = torch.tensor( [lk] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)                 
            k = torch.cat([u[:v] for u, v in zip(k, k_lens)])
            v = torch.cat([u[:v] for u, v in zip(v, k_lens)])
            q = q.reshape(-1, *q.shape[-2:])
            q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
            cu_seqlens_q=torch.cat([k_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        else:
            cu_seqlens_q = torch.tensor([0, lq], dtype=torch.int32, device="cuda")
            cu_seqlens_k = torch.tensor([0, lk], dtype=torch.int32, device="cuda")
            q = q.squeeze(0)
            k = k.squeeze(0)
            v = v.squeeze(0)


    # apply attention
    if attn=="sage":
        x = sageattn_varlen_wrapper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_kv= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_kv=lk,
        ).unflatten(0, (b, lq))
    elif attn=="sage2":
        import math
        if cross_attn or True:
            qkv_list = [q,k,v]
            del q,k,v

            x = sageattn_wrapper(qkv_list, lq) #.unsqueeze(0)
        # else:
        #     layer =  offload.shared_state["layer"]
        #     embed_sizes = offload.shared_state["embed_sizes"] 
        #     current_step = offload.shared_state["step_no"] 
        #     max_steps = offload.shared_state["max_steps"]  


        #     nb_latents =  embed_sizes[0] * embed_sizes[1]* embed_sizes[2]

        #     window = 0
        #     start_window_step = int(max_steps * 0.3)
        #     start_layer = 10
        #     end_layer = 30
        #     if (layer < start_layer or layer > end_layer )  or current_step <start_window_step: 
        #         window = 0
        #     else:
        #         # coef =  min((max_steps - current_step)/(max_steps-start_window_step),1)*max(min((25 - layer)/(25-start_layer),1),0) * 0.7 + 0.3
        #         coef = 0.3
        #         print(f"step: {current_step}, layer: {layer}, coef:{coef:0.1f}]")
        #         window =  math.ceil(coef* nb_latents)

        #     invert_spaces = (layer + current_step) % 2 == 0 and window > 0
        #     invert_spaces = False
        #     def flip(q):
        #         q = q.reshape(*embed_sizes, *q.shape[-2:])
        #         q = q.transpose(0,2)
        #         q = q.contiguous()
        #         q = q.transpose(0,2)
        #         q = q.reshape( -1, *q.shape[-2:])
        #         return q

        #     def flop(q):
        #         q = q.reshape(embed_sizes[2], embed_sizes[1], embed_sizes[0] , *q.shape[-2:])
        #         q = q.transpose(0,2)
        #         q = q.contiguous()
        #         q = q.transpose(0,2)
        #         q = q.reshape( -1, *q.shape[-2:])
        #         return q


        #     if invert_spaces:

        #         q = flip(q)
        #         k = flip(k)
        #         v = flip(v)            
        #     qkv_list = [q,k,v]
        #     del q,k,v



        #     x = sageattn_window_wrapper(qkv_list, lq, window= window) #.unsqueeze(0)

        #     if invert_spaces:
        #         x = flop(x)
        #     x = x.unsqueeze(0)

        
    elif attn=="sdpa":
        qkv_list = [q, k, v]
        del q ,k ,v
        x = sdpa_wrapper( qkv_list, lq) #.unsqueeze(0)
    elif attn=="flash" and version == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif attn=="flash":
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output

    elif attn=="xformers":
        from xformers.ops.fmha.attn_bias import BlockDiagonalPaddedKeysMask
        if b != 1 and k_lens != None:
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([lq] * b , lk, list(k_lens) ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        else:
            x = memory_efficient_attention(q, k, v )
    
    return x.type(out_dtype)