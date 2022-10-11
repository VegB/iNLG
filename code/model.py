import pdb
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Tuple, Optional
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Model, T5ForConditionalGeneration, T5EncoderModel, T5DenseReluDense, T5DenseGatedGeluDense, T5Attention

import utils


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class iNLGWrapper(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "lm"

    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.lm = self.args.lm_class(config)
    
    def _init_weights(self, module):
        if self.args.model_type.find('bart') > -1:
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_bart.py#L486
            std = self.config.init_std
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        elif self.args.model_type.find('t5') > -1:
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/t5/modeling_t5.py#L757
            factor = self.config.initializer_factor  # Used for testing weights initialization
            if isinstance(module, T5LayerNorm):
                module.weight.data.fill_(factor * 1.0)
            elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
                module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            elif isinstance(module, T5DenseReluDense):
                module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                if hasattr(module.wi, "bias") and module.wi.bias is not None:
                    module.wi.bias.data.zero_()
                module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
                if hasattr(module.wo, "bias") and module.wo.bias is not None:
                    module.wo.bias.data.zero_()
            elif isinstance(module, T5DenseGatedGeluDense):
                module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                    module.wi_0.bias.data.zero_()
                module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                    module.wi_1.bias.data.zero_()
                module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
                if hasattr(module.wo, "bias") and module.wo.bias is not None:
                    module.wo.bias.data.zero_()
            elif isinstance(module, T5Attention):
                d_model = self.config.d_model
                key_value_proj_dim = self.config.d_kv
                n_heads = self.config.num_heads
                module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
                module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
                module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
                module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
                if module.has_relative_attention_bias:
                    module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5)) 
    
    def _normalize(self, feature):
        norm = feature.norm(p=2, dim=1, keepdim=True)
        feature = feature.div(norm + 1e-16)
        return feature


class ClipCaptionModel(iNLGWrapper):

    def __init__(self, config, args):
        super().__init__(config, args)

        self.main_input_name = 'input_ids'
        self.prefix_size = self.args.visual_prefix_dim
        self.prefix_length = self.args.visual_prefix_length
        self.clip_length = self.args.visual_prefix_clip_length
        self.transformer_layers = 8
        self.lm_embedding_size = 768 if self.args.model_type != 'bart-large' else 1024

        # map visual features to input embeddings
        if self.args.mapper_type == 'mlp':
            self.clip_project = MLP((self.prefix_size, (self.lm_embedding_size * self.prefix_length) // 2,
                                     self.lm_embedding_size * self.prefix_length))
        elif self.args.mapper_type == 'transformer':
            self.clip_project = TransformerMapper(
                self.prefix_size, self.lm_embedding_size, self.prefix_length, self.clip_length, self.transformer_layers)

        # project lm hidden states to clip space
        if self.args.model_type in ['gpt2']:
            self.hidden_dim = self.config.n_embd
        else:
            self.hidden_dim = self.config.d_model
        self.clip_emb_dim = 512 if self.args.clip_features == 'vitb32' else 640
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.clip_emb_dim))

        self.init_weights()

    def assign_embedder(self):
        # Word Embedder
        if self.args.model_type in ['bart-base', 'bart-large']:
            self.embedder = self.lm.model.encoder.embed_tokens
        elif self.args.model_type in ['gpt2']:
            self.embedder = self.lm.transformer.wte
        elif self.args.model_type in ['t5-base', 't5-large']:
            self.embedder = self.lm.shared

    def _project_image_features(self, images):
        image_feat_size = images.size()
        if len(image_feat_size) == 2:
            return self.clip_project(images).view(-1, self.prefix_length, self.lm_embedding_size)
        batch_size, _, feat_dim = image_feat_size
        return self.clip_project(images[:, 1:, :].reshape(-1, feat_dim)).view(batch_size, -1, self.lm_embedding_size)

    def _concat_image_text_embeddings(self, images, input_ids):
        projected_prefix_embedding = self._project_image_features(images)
        text_embedding = self.embedder(input_ids) 
        concatenated_embedding = torch.cat((projected_prefix_embedding, text_embedding), dim=1)
        return concatenated_embedding

    def forward(self, input_ids=None, attention_mask=None, images=None, labels=None):
        if self.args.model_type in ['gpt2']:
            dummy_token = utils.get_dummy_token(input_ids.shape[0], self.prefix_length, input_ids.device)
            labels = torch.cat((dummy_token, labels), dim=1)

        outputs = self.lm(
            input_ids=None,
            inputs_embeds=self._concat_image_text_embeddings(images=images, input_ids=input_ids),
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if self.args.model_type in ['gpt2']:
            outputs.logits = outputs.logits[:, self.prefix_length-1:-1]
            decoder_hidden_states = outputs.hidden_states[0][:, self.prefix_length-1:-1]
        else:
            decoder_hidden_states = outputs.decoder_hidden_states[-1]
        projected_pred_feat = self._normalize(self.projection(decoder_hidden_states.mean(dim=1)))
        outputs.projected_pred_feat = projected_pred_feat.half()
        return outputs

    def generate(self, input_ids=None, attention_mask=None, images=None, labels=None, 
                 num_beams=None, max_length=None, synced_gpus=None):
        if self.args.model_type in ['gpt2']:
            dummy_token = utils.get_dummy_token(input_ids.shape[0], self.prefix_length, input_ids.device)
            labels = torch.cat((dummy_token, labels), dim=1)
            outputs = self.lm(
                input_ids=None,
                inputs_embeds=self._concat_image_text_embeddings(images=images, input_ids=input_ids),
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, self.prefix_length-1:-1]
            tokens = torch.argmax(logits, -1)
            return tokens

        if self.args.beam_search:
            return self.lm.generate(
                input_ids=None,
                inputs_embeds=self._concat_image_text_embeddings(images=images, input_ids=input_ids),
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
            )
        else:
            return self.lm.generate(
                input_ids=None,
                inputs_embeds=self._concat_image_text_embeddings(images=images, input_ids=input_ids),
                attention_mask=attention_mask,
                do_sample=True,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                max_length=max_length,
                num_return_sequences=1,
            )


class ContrastiveModel(iNLGWrapper):

    def __init__(self, config, args):
        super().__init__(config, args)

        self.hidden_dim = self.config.d_model
        self.clip_emb_dim = 512 if self.args.clip_features == 'vitb32' else 768
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.clip_emb_dim))
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, images=None, labels=None):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        decoder_hidden_states = outputs.decoder_hidden_states[-1]
        projected_pred_feat = self._normalize(self.projection(decoder_hidden_states.mean(dim=1)))
        outputs.projected_pred_feat = projected_pred_feat.half()
        return outputs

    def generate(self, input_ids=None, attention_mask=None, images=None, labels=None, 
                 num_beams=None, max_length=None, synced_gpus=None):
        return self.lm.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
        )
