# from dataclasses import dataclass, field
# from fairseq.dataclass import FairseqDataclass, ChoiceEnum
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Tuple
# from fairseq.modules.layer_norm import Fp32LayerNorm
# from fairseq.modules.transpose_last import TransposeLast
# from fairseq.modules.group_norm import Fp32GroupNorm
# from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, TransformerSentenceEncoderLayer, ConformerWav2Vec2EncoderLayer
# @dataclass
# class LFP2VecConfig(FairseqDataClass):
#     extractor_mode: EXTRACTOR_MODE_CHOICES = field(
#         default="default",
#         metadata={
#             "help": "mode for feature extractor. default has a single group norm with d "
#             "groups in the first conv block, whereas layer_norm has layer norms in "
#             "every block (meant to use with normalize=True)"
#         },  
#     )
#     encoder_layers: int = field(
#         default=12, metadata={"help": "num encoder layers in the transformer"}
#     )
#     encoder_embed_dim: int = field(
#         default=768, metadata={"help": "encoder embedding dimension"}
#     )
#     encoder_ffn_embed_dim: int = field(
#         default=3072, metadata={"help": "encoder embedding dimension for the feedforward network"}
#     )
#     encoder_attention_heads: int = field(
#         default=8, metadata={"help": "num encoder attention heads"}
#     )
#     attention_dropout: float = field(
#         default=0.1, metadata={"help": "dropout probability for attention weights"}
#     )
#     activation_dropout: float = field(
#         default=0.1, metadata={"help": "dropout probability for activation"}
#     )
#     activation_fn: str = field(
#         default="relu", metadata={"help": "activation function to use"}
#     )
#     layer_norm_first: bool = field(
#         default=False, metadata={"help": "apply layernorm first in the transformer blocks"}
#     )
#     layer_type: str = field(
#         default="transformer", metadata={"help": "type of layer to use in the transformer"}
#     )
#     depthwise_conv_kernel_size: int = field(
#         default=31, metadata={"help": "kernel size for the depthwise convolution"}
#     )
#     attn_type: str = field(
#         default="full", metadata={"help": "type of attention to use in the transformer"}
#     )
#     fp16: bool = field(
#         default=False, metadata={"help": "use fp16 for the transformer"}
#     )
#     required_seq_len_multiple: int = field(
#         default=2, metadata={"help": "required sequence length multiple"}
#     )
#     adapter_num: int = field(
#         default=201, metadata={"help": "number of adapters"}
#     )
#     adapter_dim: int = field(
#         default=64, metadata={"help": "dimension of the adapter"}
#     )
#     adapter_act_fn: str = field(
#         default="relu", metadata={"help": "activation function to use in the adapter"}
#     )
#     adapter_num: int = field(
#         default=201, metadata={"help": "number of adapters"}
#     )
#     adapter_dim: int = field(
#         default=64, metadata={"help": "dimension of the adapter"}
#     )
#     adapter_act_fn: str = field(
#         default="relu", metadata={"help": "activation function to use in the adapter"}
#     )   
#     dropout: float = field(
#         default=0.1, metadata={"help": "dropout probability for the transformer"}
#     )
#     required_seq_len_multiple: int = field(
#         default=2, metadata={"help": "required sequence length multiple"}
#     )
#     conv_bias: bool = field(
#         default=False, metadata={"help": "use bias in the conv layers"}
#     )
#     mode: str = field(
#         default="default", metadata={"help": "mode for the conv layers"}
#     )
#     logit_temp: float = field(
#         default=0.1, metadata={"help": "temperature to divide logits by"}
#     )
#     quantize_targets: bool = field(
#         default=False, metadata={"help": "use quantized targets"}
#     )
#     quantize_input: bool = field(
#         default=False, metadata={"help": "use quantized inputs"}
#     )
#     conv_feature_layers: str = field(
#         default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
#         metadata={
#             "help": "string describing convolutional feature extraction layers in form of a python list that contains "
#             "[(dim, kernel_size, stride), ...]"
#         },
#     )
   

# class ConvFeatureExtractor(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
#         def __init__(self, 
#                      convLayers: List[Tuple[int, int, int]],
#                      dropOut: float = 0.0,
#                      conv_bias: bool = False,
#                      mode: str = "default"):
#             super().__init__()
#             assert mode in {"default", "layer_norm"}
#             def block_creation(in_channels,
#                                out_channels,
#                                kernel_size,
#                                stride,
#                                is_layer_norm, 
#                                is_group_norm, 
#                                conv_bias):
#                 def make_conv_layer():
#                     conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=conv_bias)
#                     nn.init.kaiming_normal_(conv.weight)
#                     return conv
#                 if is_layer_norm:
#                     return nn.Sequential(
#                         make_conv_layer(in_channels, out_channels, kernel_size, stride, is_layer_norm, is_group_norm, conv_bias),
#                         nn.Dropout(p=dropOut),
#                         nn.Sequential(
#                             TransposeLast(),
#                             Fp32LayerNorm(dim, elementwise_affine=True), ## find out what is dim and how it is used
#                             TransposeLast(),
#                         ),
#                         nn.GELU(),
#                     )
#                 elif is_group_norm:
#                     return nn.Sequential(
#                         make_conv_layer(in_channels, out_channels, kernel_size, stride, is_layer_norm, is_group_norm, conv_bias),
#                         nn.Dropout(p=dropOut),
#                         nn.Sequential(
#                             TransposeLast(),
#                             Fp32GroupNorm(dim, elementwise_affine=True),
#                             TransposeLast(),
#                             )
#                         )
#                 else:
#                     return nn.Sequential(
#                         make_conv_layer(in_channels, out_channels, kernel_size, stride, is_layer_norm, is_group_norm, conv_bias),
#                         nn.Dropout(p=dropOut),
#                         nn.GELU()
#                     )
#             in_d = k  # number of input channels i the  signal , we have this values as 1 in wave2vec2 as the audio signals are purley  1D
#             self.conv_layers = nn.ModuleList()
#             for i, cl in enumerate(convLayers):
#                 assert len(cl) == 3, "invalid conv definition: " + str(cl)
#                 (dim, k, stride) = cl
#                 self.conv_layers.append(
#                     block_creation(in_d, dim, k, stride, is_layer_norm, is_group_norm, conv_bias)
#                 )
#                 in_d = dim
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         for conv in self.con_layers:
#             x = conv(x)
#         return x

# ## This layer is suppose to embed the position of the probe location for different probes in the latent space
# class ProbeEmbeddingLayer(nn.Module):
#     def __init__(self, embedding_dim, num_classes):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.embedding = nn.Embedding(num_classes, embedding_dim)
#         self.projection = nn.Linear(embedding_dim, embedding_dim)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.projection(x)
#         return x
        



# class TransformerEncoder(nn.Module):
#     def build_encoder_layer(self, args: Wav2Vec2Config, **kwargs):
#         if args.layer_type == "transformer":
#             layer = TransformerSentenceEncoderLayer(
#                 embedding_dim=self.embedding_dim,
#                 ffn_embedding_dim=args.encoder_ffn_embed_dim,
#                 num_attention_heads=args.encoder_attention_heads,
#                 dropout=self.dropout,
#                 attention_dropout=args.attention_dropout,
#                 activation_dropout=args.activation_dropout,
#                 activation_fn=args.activation_fn,
#                 layer_norm_first=args.layer_norm_first,
#             )
#         elif args.layer_type == "conformer":
#             layer = ConformerWav2Vec2EncoderLayer(
#                 embed_dim=self.embedding_dim,
#                 ffn_embed_dim=args.encoder_ffn_embed_dim,
#                 attention_heads=args.encoder_attention_heads,
#                 dropout=args.dropout,
#                 depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
#                 activation_fn="swish",
#                 attn_type=args.attn_type,
#                 use_fp16=args.fp16,
#                 pos_enc_type="abs",
#             )
#         elif args.layer_type == "trf_adp":
#             layer = TransformerSentenceEncoderLayer(
#                 embedding_dim=self.embedding_dim,
#                 ffn_embedding_dim=args.encoder_ffn_embed_dim,
#                 num_attention_heads=args.encoder_attention_heads,
#                 dropout=self.dropout,
#                 attention_dropout=args.attention_dropout,
#                 activation_dropout=args.activation_dropout,
#                 activation_fn=args.activation_fn,
#                 layer_norm_first=args.layer_norm_first,
#             )
#         else:
#             raise ValueError(f"Unsupported layer type: {args.layer_type}")
    

            
            
            
            