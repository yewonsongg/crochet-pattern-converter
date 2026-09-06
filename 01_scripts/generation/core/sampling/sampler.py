from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    case_id: str | None = None,
  ) -> SampledParameters:
    """Sample one concrete instance of the configured class.

    Explicit overrides are applied as part of the sampling request and should be incorporated before dependent parameters are sampled or derived. For example, overriding a ring variant to ``chain`` should activate the chain-specific parameters before deriving its diameter.

    Args:
      rng: Random-number generator used for stochastic sampling.
      seed: Optional seed recorded in sampling provenance.
      overrides: Optional mapping of parameter names to forced values. The supplied values are recorded in sampling provenance.
      case_id: Optional identifier for a coverage, rendering, smoke-test, or regression case.

    Returns:
      Sampled parameters, derived values, topology, components and provenance for one class instance.    

    Raises:
      KeyError: If an override references an unknown parameter or if a required conditional case is unavailable.
      ValueError: If sampling dependencies cannot be resolved or an override is invalid.
    """

    applied_overrides = dict(overrides or {})

    result = sample_class(
      spec = self.spec, 
      rng = rng,
      overrides = overrides,
    )

    provenance = create_sampling_provenance(
      config_identity = self.config_identity, 
      class_group = self.spec.class_group, 
      class_name = self.spec.class_name, 
      seed = seed, 
      decisions = result.decisions, 
      parameters = result.parameters, 
      derived = result.derived,
      overrides = applied_overrides,
      case_id = case_id,
    )

    return SampledParameters(
      parameters = result.parameters, 
      derived = result.derived, 
      topology = result.topology, 
      components = result.components, 
      provenance = provenance
    )
