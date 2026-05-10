from __future__ import annotations

from dataclasses import dataclass

from amfd.core.models import SensorWindow


@dataclass(frozen=True)
class AugmentationRequest:
    fault_label: str
    samples: int
    intensity: float


class SyntheticFaultAugmentor:
    """Interface for future VAE-WGAN/RL augmentation experiments."""

    def generate(self, request: AugmentationRequest) -> list[SensorWindow]:
        raise NotImplementedError(
            "Connect this hook to a trained VAE-WGAN or policy-driven augmenter."
        )
