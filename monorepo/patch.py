import torch
from ldm_patched.k_diffusion.sampling import BatchedBrownianTree



class BrownianTreeNoiseSamplerPatched:
    transform = None
    tree = None

    @staticmethod
    def global_init(x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        if ldm_patched.modules.model_management.directml_enabled:
            cpu = True

        t0, t1 = transform(torch.as_tensor(sigma_min)), transform(torch.as_tensor(sigma_max))

        BrownianTreeNoiseSamplerPatched.transform = transform
        BrownianTreeNoiseSamplerPatched.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __call__(sigma, sigma_next):
        transform = BrownianTreeNoiseSamplerPatched.transform
        tree = BrownianTreeNoiseSamplerPatched.tree

        t0, t1 = transform(torch.as_tensor(sigma)), transform(torch.as_tensor(sigma_next))
        return tree(t0, t1) / (t1 - t0).abs().sqrt()
