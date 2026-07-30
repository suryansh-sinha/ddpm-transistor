import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):

    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim    # Sinusoidal time embedding dimension.

        # 2 Layer MLP
        self.linear1 = nn.Linear(time_emb_dim, time_emb_dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_emb_dim * 4, time_emb_dim * 4)

    def sinusoidal_time_embedding(self, time_steps: torch.Tensor):
        """
            The even odd rule is for the dimension index i and not the timestep t.
            for each timestep t, we calculate the sin and cos in alternate manner
            pos 0 gets sin, pos 1 gets cos, pos 2 gets sin, pos 3 gets cos ... 
            The formula is - sin(t / 10000^(2i/d_model)) for 2i and cos(t / 10000^(2i/d_model)) for 2i+1.
            Here, i ranges from 0 to time_emb_dim // 2 
            Inputs:|
                time_steps (torch.Tensor): Batched integer timesteps for images of shape [B,]
            Outputs:
                emb (torch.Tensor): batched embedding for input timesteps, has shape [B, time_emb_dim]
        """
        # Since the argument inside the sin or cos function --> (t / 10000^(2i/d_model))
        # is basically just (t * frequency). We already have t so we need to calculate frequency part that depends on i.
        frequencies = 1 / (10000 ** (torch.arange(start=0, end=self.time_emb_dim//2, device=time_steps.device) / (self.time_emb_dim // 2)))   # Shape [time_emb_dim/2, ]

        # Now we multiply the frequencies with time_steps t. [B, ] x [time_emb_dim/2, ]
        args = time_steps.unsqueeze(1) * frequencies.unsqueeze(0)

        # Now applying sin and cos
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)

        # Now need to join these in alternating manner (sin cos sin cos ...)
        emb = torch.stack((sin_emb, cos_emb), dim=2).reshape((time_steps.shape[0], -1))

        return emb

    def forward(self, time_steps):
        time_emb = self.sinusoidal_time_embedding(time_steps)
        return self.linear2(self.act(self.linear1(time_emb)))

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups):
        r"""
        time_emb_dim: Expects sinusoidal_time_emb_dim * 4
        """
        super().__init__()
        
        self.act = nn.SiLU()

        if in_channels % num_groups != 0 or out_channels % num_groups != 0:
            raise ValueError("in_channels/out_channels must be divisible by num_groups [Res_Block]")
        
        # To map sinusoidal time embedding to out_channels
        self.lin = nn.Linear(time_emb_dim, out_channels)
        # Define First Conv Group
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Define Second Conv Group
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # For shortcut connection
        if in_channels != out_channels:
            self.linearProj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.linearProj = nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.lin(self.act(time_emb))
        x1 = self.conv1(self.act(self.norm1(x)))   # First conv step complete
        x1 = x1 + time_emb[..., None, None]        # Time embedding injection complete
        x2 = self.conv2(self.act(self.norm2(x1)))  # Second conv step complete
        x2 = x2 + self.linearProj(x)               # Shortacut connection complete
        return self.act(x2)

class AttentionBlock(nn.Module):

    def __init__(self, input_dim, qkv_dim, n_heads, num_groups):
        """Takes in feature maps of shape (B, C, H, W) converts to (B, C, H*W) then H*W part acts as sequence length
        and number of channels act as embedding dimension for the input to the attention mechanism. It applies multi-head attention
        to our batch and returns outputs in (B, C, H, W) format again.
        Args:
            input_dim: Number of channels/feature_maps in input img (acts as input_dim before we convert to qkv)
            qkv_dim: This is basically d_model or the dimension of Q, K, V
            n_heads: Number of heads in multi-head self attn
            num_groups: Number of groups for GroupNorm. (input_dim must be divisible by num_groups)
        """
        super().__init__()
        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads
        self.head_dim = qkv_dim // n_heads
        self.qkv_proj = nn.Linear(input_dim, qkv_dim * 3) # Converting to Q, K, V in single linear layer for efficiency
        self.final_proj = nn.Linear(qkv_dim, input_dim)
        if input_dim % num_groups != 0:
            raise ValueError("input_dim must be divisible by num_groups [Attention_Block]")
        self.norm = nn.GroupNorm(num_groups, input_dim)

    # def scaled_dot_product(self, q, k, v):
    #     # Softmax(((q @ k_T) // d_k) @ v)
    #     # Inputs are -> [B, n_heads, seq_len(H*W), head_dim]
    #     sqrt_d_k = math.sqrt(q.shape[-1])
    #     scaled = q @ k.transpose(-1, -2) / sqrt_d_k # [B, n_heads, seq_len, head_dim] @ [B, n_heads, head_dim, seq_len] --> [B, n_heads, seq_len, seq_len]        
    #     attn_score = torch.softmax(scaled, dim=-1)  #[B, n_heads, seq_len, seq_len]
    #     return attn_score @ v   # [B, n_heads, seq_len, seq_len] @ [B, n_heads, seq_len, head_dim] --> [B, n_heads, seq_len, head_dim]

    def forward(self, fmap):
        # reshape from [B, C, H, W] to [B, C(input_dim), H*W (seq_len)]
        seq = torch.flatten(self.norm(fmap), start_dim=2, end_dim=-1).permute(0, 2, 1)  # [B, seq_len(H*W), input_dim(C)]
        B, seq_len, _ = seq.shape
        qkv = self.qkv_proj(seq)   # [B, seq_len, input_dim] -> [B, seq_len, qkv_dim * 3]
        qkv = qkv.reshape(B, seq_len, 3, self.n_heads, self.head_dim)   # 3 in reshape for Q,K,V
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [3, B, n_heads, seq_len, head_dim]
        q, k, v = qkv # Q, K, V have shape [B, n_heads, seq_len, head_dim]
        # Calculate attn
        out = F.scaled_dot_product_attention(q, k, v)  # [B, n_heads, seq_len, head_dim]
        # Combine all heads
        out = out.permute(0, 2, 1, 3)   # [B, seq_len, n_heads, head_dim]
        out = out.reshape(B, seq_len, self.n_heads * self.head_dim)   # [B, seq_len, qkv_dim]
        # Project to single output
        out = self.final_proj(out)  # [B, seq_len(H*W), qkv_dim] -> [B, seq_len(H*W), input_dim(C)]
        # Go back to [B, C(input_dim), H*W] format
        out = out.permute(0, 2, 1)
        out = out.reshape(fmap.shape)   # [B, C, H*W] -> [B, C, H, W]
        return out + fmap   # shortcut connection

# Reducing repeatability
class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim, num_groups)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim, num_groups)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb):
        e = self.res2(self.res1(x, time_emb), time_emb)
        return self.downsample(e), e    # returning e for skip connection
    
class MidBlock(nn.Module):

    def __init__(self, channels, qkv_dim, n_heads, time_emb_dim, num_groups):
        super().__init__()
        self.res = ResBlock(channels, channels, time_emb_dim, num_groups)
        self.attn = AttentionBlock(channels, qkv_dim, n_heads, num_groups)

    def forward(self, x, time_emb):
        return self.attn(self.res(x, time_emb))
    
class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.res2 = ResBlock(in_channels*2, in_channels, time_emb_dim, num_groups)
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim, num_groups)

    def forward(self, x, time_emb, skip_features):
        x = self.upsample(x)
        x = torch.concat((x, skip_features.pop()), dim=1)
        x = self.res1(self.res2(x, time_emb), time_emb)
        return x


# The UNet Class
class UNet(nn.Module):

    def __init__(self, in_channels, base_channels, sinusoidal_time_emb_dim, qkv_dim, n_heads, num_groups, depth):
        
        super().__init__()
        channel_prog = [base_channels * (2**(i)) for i in range(depth)]  # [32, 64, 128, 256]

        # Define the stem
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Define time embedding
        self.time_emb_layer = TimeEmbedding(sinusoidal_time_emb_dim)  # Outputs embedding -> [B, self.time_emb_dim * 4]
        time_emb_dim = sinusoidal_time_emb_dim * 4
        
        # Define the encoder blocks
        current_channels = base_channels
        self.encoder_blocks = nn.ModuleList([])
        for i in channel_prog:
            self.encoder_blocks.append(DownBlock(current_channels, i, time_emb_dim, num_groups))
            current_channels = i
        

        # Define the bottleneck blocks
        self.mid_blocks = nn.ModuleList([
            MidBlock(current_channels, qkv_dim, n_heads, time_emb_dim, num_groups),
            MidBlock(current_channels, qkv_dim, n_heads, time_emb_dim, num_groups),
        ])

        # Define the decoder blocks
        self.decoder_blocks = nn.ModuleList([])
        for i in range(len(channel_prog)-1, -1, -1):
            if i == 0:
                self.decoder_blocks.append(UpBlock(channel_prog[i], base_channels, time_emb_dim, num_groups))
            else:
                self.decoder_blocks.append(UpBlock(channel_prog[i], channel_prog[i-1], time_emb_dim, num_groups))

        # Final Conv to convert from base_channels to rgb channels
        self.linearOut = nn.Conv2d(base_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time_steps):

        time_emb = self.time_emb_layer(time_steps)
        x = self.stem(x)

        skip_features = []
        for block in self.encoder_blocks:
            x, skip = block(x, time_emb)
            skip_features.append(skip)

        for block in self.mid_blocks:
            x = block(x, time_emb)

        for block in self.decoder_blocks:
            x = block(x, time_emb, skip_features)

        x = self.linearOut(x)
        return x