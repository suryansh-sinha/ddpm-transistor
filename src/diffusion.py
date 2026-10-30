import torch


class LinearNoiseScheduler:
    def __init__(
        self, num_timesteps: int, beta_start: float, beta_end: float, device: str
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # All these tensors have dimension --> [1000] if num_timesteps = 1000
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas.to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars).to(device)

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        # sqrt_alpha_bars[t] has shape [batch_size] and we need to multiply it with [batch_size, 3, 256, 256] image.
        # for broadcasting to work correctly, we are adding dimensions to make it [batch_size, 1, 1, 1]

        batch_size = x_0.shape[0]

        sqrt_alpha_bars = (
            self.sqrt_alpha_bars[t].reshape([batch_size, 1, 1, 1]).to(self.device)
        )  # Bx1x1x1
        sqrt_one_minus_alpha_bars = (
            self.sqrt_one_minus_alpha_bars[t]
            .reshape([batch_size, 1, 1, 1])
            .to(self.device)
        )  # Bx1x1x1
        x_t = sqrt_alpha_bars * x_0 + sqrt_one_minus_alpha_bars * noise

        # the direct formula to get x_t from x_0
        return x_t

    def sampling(self, unet: torch.nn.Module, x_t: torch.Tensor, starting_step=None):

        unet.eval()
        batch_size, img_channels, img_size = x_t.shape[0], x_t.shape[1], x_t.shape[2]

        if starting_step is None:
            starting_step = self.num_timesteps

        for t in range(starting_step - 1, -1, -1):
            if (t + 1) % 100 == 0:
                print(f"Working on timestep: {t + 1}/{self.num_timesteps}")

            # t is an integer. Want make tensor that contains t with shape -> [batch_size]
            T = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            # T = torch.tensor([t]).expand(batch_size).to(self.device)

            sigma_1 = torch.sqrt(self.betas[T]).to(
                self.device
            )  # fixed large variance (choice 1 in paper)
            # sigma_2 = (self.sqrt_one_minus_alpha_bars[T-1]/self.sqrt_one_minus_alpha_bars[T]) * torch.sqrt(self.betas[T])   # tighter posterior variance (choice 2)

            noise = sigma_1.reshape(-1, 1, 1, 1) * torch.randn(
                size=(batch_size, img_channels, img_size, img_size), device=self.device
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unet_pred_noise = unet(x_t, T)

            # Rearrange x_t = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar * noise so x0 is on LHS.
            # eqn 15 in paper
            estimated_x0 = (1 / torch.sqrt(self.alpha_bars[T])).reshape(-1, 1, 1, 1) * (
                x_t
                - (
                    self.sqrt_one_minus_alpha_bars[T].reshape(-1, 1, 1, 1)
                    * unet_pred_noise
                )
            )
            estimated_x0 = torch.clamp(
                estimated_x0, -1.0, 1.0
            )  # Since clean img can only have values in range (-1, 1)

            if t != 0:
                # Calculate true mathematical posterior mean using eqn 7
                mean = (
                    (self.sqrt_alpha_bars[T - 1] * self.betas[T])
                    / (1.0 - self.alpha_bars[T])
                ).reshape(-1, 1, 1, 1) * estimated_x0 + (
                    (torch.sqrt(self.alphas[T]) * (1.0 - self.alpha_bars[T - 1]))
                    / (1.0 - self.alpha_bars[T])
                ).reshape(-1, 1, 1, 1) * x_t
            else:
                mean = estimated_x0
                noise = torch.zeros_like(noise).to(self.device)

            x_t = (
                mean + noise
            )  # x_t-1 = posterior mean (depends on estimated x0, xt) + variance * noise from normal dist

        return x_t  # (B, C, H, W)
