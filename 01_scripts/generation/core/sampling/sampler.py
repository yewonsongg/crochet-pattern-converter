from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import ConfigIdentity, SampledParameters
from .provenance import create_sampling_provenance
from .resolver import sample_class
from .schema import ClassSpec


@dataclass(frozen=True)
class ClassSampler:
  """Reusable sampler bound to one resolved class specification.

  A ``ClassSampler`` samples concrete parameters for one configured class and attaches configuration and sampling provenance to each result.

  Attributes:
    spec: Resolved specification for the configured class.
    config_identity: Identity of the configuration used to create this sampler.  
  """

  spec: ClassSpec
  config_identity: ConfigIdentity

  def sample(
    self, 
    rng: np.random.Generator, 
    *, 
    seed: int | None = None
  ) -> SampledParameters:
    """Sample one concrete instance of the configured class.

    Args:
      rng: Random-number generator used for stochastic sampling.
      seed: Optional seed recorded in sampling provenance.

    Returns:
      Sampled parameters, derived values, topology, components and provenance for one class instance.    
    """

    result = sample_class(
      spec = self.spec, 
      rng = rng
    )

    provenance = create_sampling_provenance(
      config_identity = self.config_identity, 
      class_group = self.spec.class_group, 
      class_name = self.spec.class_name, 
      seed = seed, 
      decisions = result.decisions, 
      parameters = result.parameters, 
      derived = result.derived,
    )

    return SampledParameters(
      parameters = result.parameters, 
      derived = result.derived, 
      topology = result.topology, 
      components = result.components, 
      provenance = provenance
    )