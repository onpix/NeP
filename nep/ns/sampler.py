from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union
import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples


class GaussianProposalSampler(ProposalNetworkSampler):
    def generate_ray_samples(
            self,
            ray_bundle: Optional[RayBundle] = None,
            density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(
            self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[
                i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(
                    ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(
                    ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    # modify: taking samples as input instead of positions
                    density = density_fns[i_level](ray_samples)
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples)
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
