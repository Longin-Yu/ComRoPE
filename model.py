# I don't know why this implementation is slower.


from typing import Optional, Tuple, List
import torch
from torch import nn
from torchvision.transforms import transforms, InterpolationMode

from transformers import CLIPVisionModel, CLIPVisionConfig, PretrainedConfig, PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPAttention, logger
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutputWithNoAttention

# recommended to use torch.matrix_exp instead of naive_matrix_exp
def naive_matrix_exp(A: torch.Tensor, num_order=32):
    A_k = torch.eye(A.shape[-1]).to(dtype=A.dtype, device=A.device)
    A_k = A_k.repeat(*(A.shape[:-2]), 1, 1)
    outputs = A_k
    factorial_value = 1.0
    for k in range(1, num_order + 1):
        A_k = torch.matmul(A_k, A)
        factorial_value *= k
        outputs += A_k / factorial_value
    return outputs

class CustomVITConfig(CLIPVisionConfig):
    def __init__(
        self, 
        custom_pe_type: str = "raw",
        init_std: float = 1.0,
        axis_scale: bool = True,
        prepend_cls: bool = True,
        cls_position: float = 0.5,
        theta_base: float = 1e-100,
        block_size: int = 2,
        scale_factor: float = 1e100,
        num_axes: int = 2,
        num_frames: int = 16, # only for num_axes = 3
        keep_ape: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_pe_type = custom_pe_type
        self.init_std = init_std
        self.axis_scale = axis_scale
        self.prepend_cls = prepend_cls
        self.cls_position = cls_position
        self.theta_base = theta_base
        self.block_size = block_size
        self.scale_factor = scale_factor
        self.num_axes = num_axes
        self.num_frames = num_frames
        self.keep_ape = keep_ape
        assert self.num_axes == 2 or self.num_axes == 3
    
    def get_positions(self, perturbation_intensity: float = 0, offset_intensity: float = 0) -> torch.Tensor:
        offset = torch.randn(self.num_axes) * offset_intensity
        num_patches_on_single_axis = self.image_size // self.patch_size
        if self.num_axes == 2:
            positions = torch.stack(torch.meshgrid(
                torch.arange(0, num_patches_on_single_axis, dtype=torch.float),
                torch.arange(0, num_patches_on_single_axis, dtype=torch.float),
                indexing="ij",
            ), dim=-1).view(-1, 2)
            perturbation = torch.randn_like(positions) * perturbation_intensity * 0.5
            perturbation = torch.clamp(perturbation, -0.5, 0.5)
            positions = positions + perturbation
            if self.axis_scale:
                positions = (positions + 0.5) / num_patches_on_single_axis
            if self.prepend_cls:
                cls_position = torch.tensor([self.cls_position, self.cls_position])
                positions = torch.cat([cls_position.unsqueeze(0), positions], dim=0)
            return positions + offset
        elif self.num_axes == 3:
            F = torch.arange(self.num_frames)
            P = torch.arange(num_patches_on_single_axis)
            grids = torch.meshgrid(F, P, P, indexing='ij')
            positions = torch.stack(grids, dim=-1).view(-1, 3).to(torch.float)
            perturbation = torch.randn_like(positions) * perturbation_intensity * 0.5
            perturbation = torch.clamp(perturbation, -0.5, 0.5)
            perturbation[:,0] = 0 # set frame perturbation = 0
            positions = positions + perturbation
            if self.axis_scale:
                positions = (positions + 0.5) / num_patches_on_single_axis
            if self.prepend_cls:
                cls_position = torch.tensor([self.cls_position, self.cls_position, self.cls_position])
                positions = torch.cat([cls_position.unsqueeze(0), positions], dim=0)
            return positions + offset
        else:
            raise NotImplementedError

class CustomVITForClassificationConfig(CustomVITConfig):
    def __init__(
        self,
        num_class: int = 1000,
        mlp_intermediate: int = 0, # zero for only a single linear layer
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_class = num_class
        self.mlp_intermediate = mlp_intermediate

class RoPESelfAttentionBase(CLIPAttention):
    block_diag_selector: torch.Tensor # (self.head_dim, self.head_dim)
    def __init__(
        self, 
        config: CustomVITConfig,
    ):
        super().__init__(config)
        
        self.positions: Optional[torch.Tensor] = None
        self.block_size: int = config.block_size
        
        assert self.head_dim % self.block_size == 0
        
        self.register_buffer("block_diag_selector", self._init_block_diag_selector(), False)
        
        self.num_axes = config.num_axes
    
    def _init_block_diag_selector(self) -> torch.Tensor:
        block = torch.ones(self.block_size, self.block_size, dtype=torch.bool)
        eye = torch.eye(self.head_dim // self.block_size, dtype=torch.bool)
        return torch.kron(eye, block)
    
    def get_exponent_matrix(self) -> torch.Tensor:
        """Get the exponet skew-symmetric matrix `A` where `R = exp(sum(Ax))` is the rotation matrix

        Returns:
            torch.Tensor: (num_heads or 1, num_axes, head_dim / block_size, block_size, block_size)
        """
        raise NotImplementedError
    
    def get_rotation_matrix(self) -> torch.Tensor:
        """Get the final rotation matrix `R = exp(sum(Ax))`

        Returns:
            torch.Tensor: (bs, len, num_heads, head_dim, head_dim)
        """
        if self.positions is None:
            raise ValueError("positions must be set before calling get_rotation_matrix")
        A = self.get_exponent_matrix() # (num_heads or 1, num_axes, head_dim / block_size, block_size, block_size)
        # sum (A positions) over all axes
        eye = torch.eye(self.block_size, device=A.device, dtype=A.dtype)
        positions = self.positions.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (bs, len, num_axes, 1, 1, 1)
        # positions = positions[:1] # (1, len, num_axes, 1, 1)
        positions = positions * eye # (bs, len, num_axes, 1, block_size, block_size)
        positions = positions.unsqueeze(2) # (bs, len, 1, num_axes, 1, block_size, block_size)
        # A: (num_heads or 1, num_axes, head_dim / block_size, block_size, block_size)
        lnR = torch.matmul(A, positions) # (bs, len, num_heads or 1, num_axes, head_dim / block_size, block_size, block_size)
        lnR = lnR.sum(dim=3) # (bs, len, num_heads or 1, head_dim / block_size, block_size, block_size)
        # R = naive_matrix_exp(lnR) # if you want to use naive matrix exp
        R = torch.matrix_exp(lnR)  # (bs, len, num_heads or 1, head_dim / block_size, block_size, block_size)
        diagR = torch.zeros(*(R.shape[:3]), self.head_dim, self.head_dim, device=A.device) # (bs, len, num_heads or 1, head_dim, head_dim)
        diagR[:,:,:,self.block_diag_selector] = R.view(*(R.shape[:3]), -1)
        return diagR 
    
    def set_positions(self, positions: torch.Tensor):
        assert len(positions.shape) == 3, f"positions must be 3D (batchsize, len, num_axes), but got {positions.shape}"
        assert positions.size(2) == self.config.num_axes, f"positions must have the same number of axes as the model, but got {positions.size(2)} and {self.config.num_axes}"
        self.positions = positions
    
    def unset_positions(self):
        self.positions = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale  # (bsz, len, embed_dim)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz) # (bsz, num_heads, len, head_dim)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz) # (bsz, num_heads, len, head_dim)

        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz) # (bsz, num_heads, len, head_dim)
        
        R = self.get_rotation_matrix() # (bs, len, num_heads, head_dim, head_dim)
        R = R.transpose(1, 2) # (bs, num_heads, len, head_dim, head_dim)
        Q = query_states.unsqueeze(-1) # (bsz, num_heads, len, head_dim, 1)
        Q = torch.matmul(R, Q) # (bsz, num_heads, len, head_dim, 1)
        Q = Q.squeeze(-1) # (bsz, num_heads, src_len, head_dim)
        K = key_states.unsqueeze(-1) # (bsz, num_heads, len, head_dim, 1)
        K = torch.matmul(R, K) # (bsz, num_heads, len, head_dim, 1)
        K = K.squeeze(-1) # (bsz, num_heads, tgt_len, head_dim)
        attn_weights = torch.matmul(Q, K.transpose(2, 3)) # (bsz, num_heads, src_len, tgt_len)
        src_len = attn_weights.size(2)
        
        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            attn_weights = attn_weights + causal_attention_mask
            
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights + 0.0
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states) # (bsz, num_heads, src_len, head_dim)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, src_len, embed_dim) # (bsz, src_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def reset_pe(self):
        pass


class VanillaRoPEAttention(RoPESelfAttentionBase):
    exponent_matrix: torch.Tensor # (1, num_axes, head_dim / 2, 2, 2)
    def __init__(
        self, 
        config: CustomVITConfig,
    ):
        super().__init__(config)
        
        if self.head_dim % (self.num_axes * 2) != 0:
            raise ValueError(f"head_dim=hidden_size/num_heads must be divisible by num_axes * 2, but got hidden_size={self.embed_dim}, head_dim={self.head_dim} and num_axes={self.num_axes}")
        
        assert self.block_size == 2
        
        self.theta_base = config.theta_base
        self.axis_rotation_dim = self.head_dim // self.num_axes
        
        exponent_matrix = self._init_exponent_matrix()
        self.register_buffer("exponent_matrix", exponent_matrix, False)
    
    def _init_exponent_matrix_legacy(self) -> torch.Tensor:
        """Return a tensor with shape (1, num_axes, head_dim, head_dim)
        """
        angle_base = self.theta_base ** torch.arange(0, self.axis_rotation_dim // 2) # (axis_rotation_dim // 2)
        basic_block = torch.tensor([[0., -1.], [1., 0.]])
        block_diag_matrix = torch.kron(torch.diag(angle_base), basic_block) # (axis_rotation_dim, axis_rotation_dim)
        ret = torch.zeros(1, self.num_axes, self.head_dim, self.head_dim)
        for axis_index in range(self.num_axes):
            start = axis_index * self.axis_rotation_dim
            end = (axis_index + 1) * self.axis_rotation_dim
            ret[0, axis_index, start:end, start:end] = block_diag_matrix
        return ret
    
    def _init_exponent_matrix(self) -> torch.Tensor:
        """Return a tensor with shape (1, num_axes, head_dim / 2, 2, 2)"""
        # Calculate the number of unique angles based on the axis rotation dimension
        num_angles = self.axis_rotation_dim // 2 # head_dim / num_axes / 2
        angle_base = self.theta_base ** torch.arange(num_angles)  # (head_dim / num_axes / 2)
        basic_block = torch.tensor([[0., -1.], [1., 0.]])
        angle_blocks = angle_base.unsqueeze(-1).unsqueeze(-1) * basic_block # (head_dim / num_axes / 2, 2, 2)

        ret = torch.zeros(1, self.num_axes, self.head_dim // 2, 2, 2)
        for axis_index in range(self.num_axes):
            start = axis_index * num_angles
            end = (axis_index + 1) * num_angles
            ret[0, axis_index, start:end] = angle_blocks
        
        return ret
    
    def get_exponent_matrix(self) -> torch.Tensor:
        return self.exponent_matrix
    


class LieREAttention(RoPESelfAttentionBase):
    def __init__(
        self, 
        config: CustomVITConfig,
    ):
        super().__init__(config)
        
        self.block_size = config.block_size
        self.init_std = config.init_std
        self.num_blocks = self.head_dim // self.block_size
        self.scale_factor = config.scale_factor
        
        if self.head_dim % self.block_size != 0:
            raise ValueError(f"head_dim must be divisible by block_size, but got {self.head_dim} and {self.block_size}")
        
        self.freqs = nn.Parameter(torch.randn(
            self.num_heads, 
            self.num_axes, 
            self.num_blocks,
            self.block_size,
            self.block_size,
        ) * (self.init_std * self.scale_factor))
        
    def get_exponent_matrix(self) -> torch.Tensor:
        skew_symmetic = self.freqs - self.freqs.transpose(-1, -2) # (num_heads, num_axes, num_blocks, block_size, block_size)
        
        skew_symmetic /= self.scale_factor
        
        return skew_symmetic

    def reset_pe(self):
        self.freqs.data.normal_(0, self.init_std * self.scale_factor)
        # eye = torch.eye(self.block_size).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # self.freqs.data[...] = eye

class ComRoPELDAttention(RoPESelfAttentionBase):
    def __init__(
        self, 
        config: CustomVITConfig,
    ):
        super().__init__(config)
        
        self.block_size = config.block_size
        self.init_std = config.init_std
        self.num_blocks = self.head_dim // self.block_size
        
        if self.head_dim % self.block_size != 0:
            raise ValueError(f"head_dim must be divisible by block_size, but got {self.head_dim} and {self.block_size}")
        
        self.freqs = nn.Parameter(torch.randn(
            self.num_heads, 
            1, 
            self.num_blocks,
            self.block_size,
            self.block_size,
        ) * self.init_std)
        self.multiplier = nn.Parameter(torch.randn(1, self.num_axes, self.num_blocks, 1, 1))
    
    def get_exponent_matrix(self) -> torch.Tensor:
        skew_symmetic = self.freqs - self.freqs.transpose(-1, -2) # (num_heads, 1, num_blocks, block_size, block_size)
        skew_symmetic = skew_symmetic * self.multiplier # (num_heads, num_axes, num_blocks, block_size, block_size)
        
        return skew_symmetic

    def reset_pe(self):
        self.freqs.data.normal_(0, self.init_std)
        self.multiplier.data.normal_(0, 1)

class ComRoPEAPAttention(RoPESelfAttentionBase):
    mask: torch.Tensor
    def __init__(
        self, 
        config: CustomVITConfig,
    ):
        super().__init__(config)
        
        self.block_size = config.block_size
        self.init_std = config.init_std
        self.axis_rotation_dim = self.head_dim // self.num_axes
        self.num_blocks = self.axis_rotation_dim // self.block_size
        self.total_blocks = self.num_blocks * self.num_axes
    
        
        if self.head_dim % self.num_axes != 0:
            raise ValueError(f"head_dim must be divisible by num_axes, but got {self.head_dim} and {self.num_axes}")
        
        if self.axis_rotation_dim % self.block_size != 0:
            raise ValueError(f"axis_rotation_dim must be divisible by block_size, but got {self.axis_rotation_dim} and {self.block_size}")
        
        self.freqs = nn.Parameter(torch.randn(
            self.num_heads, 
            1,
            self.total_blocks, # head_dim / block_size 
            self.block_size,
            self.block_size,
        ) * self.init_std)
        
        self.register_buffer("mask", self._create_mask(), False)
        
    def _create_mask(self):
        # Create a mask with shape (num_axis, total_blocks)
        mask = torch.zeros((self.num_axes, self.total_blocks), dtype=torch.float32)
        
        # Set mask[i, j] = 1 if j % num_axis == i, else 0
        for i in range(self.num_axes):
            mask[i, i::self.num_axes] = 1.0
        
        # Reshape the mask for broadcasting in (num_heads, num_axis, total_blocks, block_size, block_size)
        mask = mask.view(1, self.num_axes, self.total_blocks, 1, 1)
        return mask
    
    def get_exponent_matrix(self) -> torch.Tensor:
        skew_symmetic = self.freqs - self.freqs.transpose(-1, -2) # (num_heads, 1, total_blocks, block_size, block_size)
        
        expanded_input = skew_symmetic.expand(-1, self.num_axes, -1, -1, -1)
        
        # Multiply by the mask to get the output
        output_tensor = expanded_input * self.mask
        return output_tensor

    def reset_pe(self):
        self.freqs.data.normal_(0, self.init_std)

class CustomCLIPVisionEmbeddingsWithoutPE(nn.Module):
    def __init__(self, config: CustomVITConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(1)
        assert len(pixel_values.shape) == 5, f"Got unexpected pixel_values with shape ({pixel_values.shape})"
        # pixel_values: B, F, c, h, w
        B = pixel_values.shape[0]
        F = pixel_values.shape[1]
        pixel_values = pixel_values.flatten(end_dim=1) # BF, c, h, w
        
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds: torch.Tensor = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # BF, C, H, W
        patch_embeds = patch_embeds.view(B, F, *(patch_embeds.shape[-3:])) # B, F, C, H, W
        patch_embeds = patch_embeds.transpose(1, 2).flatten(2).transpose(1, 2) # B, L, C

        class_embeds = self.class_embedding.expand(B, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        return embeddings

class CustomCLIPVisionEmbeddings(CustomCLIPVisionEmbeddingsWithoutPE):
    def __init__(self, config: CustomVITConfig):
        super().__init__(config)

        self.num_frames = config.num_frames if config.num_axes == 3 else 1
        self.positions: Optional[torch.Tensor] = None

        self.patch_per_axis = self.image_size // self.patch_size
        self.position_embedding = nn.Parameter(
            torch.randn(self.num_frames, self.patch_per_axis, self.patch_per_axis, self.embed_dim)
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        embeddings = super().forward(pixel_values) # B, L, C
        
        if True: # Position Embeddings
            
            positions = self.positions * 2 - 1 # (batchsize, len, 2), scale to [-1, 1]
            if positions.shape[2] == 2:
                zeros = torch.zeros(positions.shape[:-1], dtype=positions.dtype, device=positions.device).unsqueeze(-1)
                positions = torch.concat((zeros, positions), dim=-1)
            pos_embeds_book = self.position_embedding # (frame, patch_per_axis, patch_per_axis, embed_dim)
            assert positions.shape[1] == embeddings.shape[1], f"positions and embeddings must have the same length, but got {positions.shape[1]} and {embeddings.shape[1]}"
            
            # interpolate the position embeddings to (batchsize, len, embed_dim)
            
            # Rearrange position embeddings for grid sampling
            pos_embeds_book = pos_embeds_book.permute(3, 0, 1, 2).unsqueeze(0)  # (1, embed_dim, frame, patch_per_axis, patch_per_axis)

            # Interpolate position embeddings based on normalized positions
            pos_embeds = torch.nn.functional.grid_sample(
                pos_embeds_book,
                positions.unsqueeze(0).unsqueeze(0), 
                align_corners=True,
                padding_mode="border",
            )
            pos_embeds = pos_embeds.squeeze(0).squeeze(1) # (embed_dim, batch_size, len)
            pos_embeds = pos_embeds.permute(1, 2, 0) # (batch_size, len, embed_dim)
            # print(pos_embeds)

        embeddings = embeddings + pos_embeds
        return embeddings

    def set_positions(self, positions: torch.Tensor):
        assert len(positions.shape) == 3, f"positions must be 3D (batchsize, len, num_axes), but got {positions.shape}"
        assert positions.size(2) == self.config.num_axes, f"positions must have the same number of axes as the model, but got {positions.size(2)} and {self.config.num_axes}"
        self.positions = positions
    
    def unset_positions(self):
        self.positions = None


class CustomVIT(CLIPVisionModel):
    config_class = CustomVITConfig
    config: CustomVITConfig
    def __init__(self, config: CustomVITConfig):
        super().__init__(config)
        
        if self.config.custom_pe_type == "raw":
            self.vision_model.embeddings = CustomCLIPVisionEmbeddings(self.config)
            
        else:
            if not self.config.keep_ape:
                self.vision_model.embeddings = CustomCLIPVisionEmbeddings(self.config)
            else:
                self.vision_model.embeddings = CustomCLIPVisionEmbeddingsWithoutPE(self.config)
            if self.config.custom_pe_type == "none":
                pass
            else:
                if config.custom_pe_type == "vanilla-rope":
                    factory = lambda: VanillaRoPEAttention(
                        config, 
                    )
                elif config.custom_pe_type == "liere":
                    factory = lambda: LieREAttention(
                        config, 
                    )
                elif config.custom_pe_type == "comrope-ld":
                    factory = lambda: ComRoPELDAttention(
                        config, 
                    )
                elif config.custom_pe_type == "comrope-ap":
                    factory = lambda: ComRoPEAPAttention(
                        config, 
                    )
                else:
                    raise ValueError(f"Unrecognized custom attention type: {config.custom_pe_type}")
                for layer in self.vision_model.encoder.layers:
                    layer.self_attn = factory()

    def custom_parameters(self) -> List[nn.Parameter]:
        params = []
        params.extend(list(self.vision_model.embeddings.parameters()))
        for layer in self.vision_model.encoder.layers:
            for name, param in layer.self_attn.named_parameters():
                if "k_proj" in name:
                    continue
                if "v_proj" in name:
                    continue
                if "q_proj" in name:
                    continue
                if "out_proj" in name:
                    continue
                params.append(param)
        return params

    def reset_pe(self, requires_grad: Optional[bool] = True):
        if self.config.custom_pe_type == "raw":
            self.vision_model.embeddings.position_embedding.reset_parameters()
            if requires_grad is not None:
                self.vision_model.embeddings.position_embedding.requires_grad_(requires_grad)
        else:
            if self.config.custom_pe_type == "none":
                pass
            else:
                for layer in self.vision_model.encoder.layers:
                    layer.self_attn.reset_pe()
                    if requires_grad is not None:
                        layer.self_attn.requires_grad_(requires_grad)
    
    def forward(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None,
        positions: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        if positions is not None:
            for layer in self.vision_model.encoder.layers:
                if hasattr(layer.self_attn, "set_positions"):
                    layer.self_attn.set_positions(positions)
            if hasattr(self.vision_model.embeddings, "set_positions"):
                self.vision_model.embeddings.set_positions(positions)
        model_outputs = super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if positions is not None:
            for layer in self.vision_model.encoder.layers:
                if hasattr(layer.self_attn, "unset_positions"):
                    layer.self_attn.unset_positions()
            if hasattr(self.vision_model.embeddings, "unset_positions"):
                self.vision_model.embeddings.unset_positions()
        return model_outputs


class CustomVITForClassification(CustomVIT):
    config_class = CustomVITForClassificationConfig
    config: CustomVITForClassificationConfig
    def __init__(self, config: CustomVITForClassificationConfig):
        super().__init__(config)
        if config.mlp_intermediate == 0:
            self.classification_head = nn.Linear(config.hidden_size, config.num_class)
        else:
            self.classification_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.mlp_intermediate),
                nn.ReLU(),
                nn.Linear(config.mlp_intermediate, config.num_class),
            )
    
    def forward(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None,
        positions: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        model_outputs = super().forward(
            pixel_values=pixel_values,
            positions=positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = model_outputs.pooler_output
        logits = self.classification_head(hidden_states)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.reset_parameters()
        else:
            super()._init_weights(module)
            
    def custom_parameters(self) -> List[nn.Parameter]:
        params = super().custom_parameters()
        params.extend(list(self.classification_head.parameters()))
        return params

class ImageTransform(transforms.Compose):
    def __init__(
        self,
        resolution: int,
        mid_resolution_ratio: float = 1.1,
    ) -> None:
        self.resolution = resolution
        self.mid_resolution = round(mid_resolution_ratio * resolution)
        super().__init__([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.mid_resolution), 
            transforms.RandomCrop((self.resolution, self.resolution)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # parameters from deit
        ])