from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..models import ConfigIdentity, SampledParameters
from .provenance import create_sampling_provenance
from .resolver import resolve_class_spec, sample_class
from .schema import ClassSpec


@dataclass(frozen=True)
class SamplingConfig:
  """Resolved sampling configuration for symbol generation.

  A ``SamplingConfig`` Stores the validated source configuration and its identity, and provides methods for resolving class specifications, constructing class samplers, and sampling concrete parameter values.

  Attributes:
    raw: Validated configuration mapping.
    identity: Version and provenance information for the configuration.
  """

  raw: Mapping[str, Any]
  identity: ConfigIdentity

  def resolve(
    self, 
    class_group: str, 
    class_name: str
  ) -> ClassSpec:
    """Resolve one class specification from the configuration.

    Args:
      class_group: Configuration group containing the class.
      class_name: Name of the class to resolve.

    Returns:
      :class:`ClassSpec`: The resolved declarative class specification.

    Raises:
      KeyError: If the group or class does not exist.
    """

    # from `resolver.py`
    return resolve_class_spec(
      raw = self.raw, 
      group = class_group, 
      name = class_name,
    )

  def sampler(
    self, 
    class_group: str, 
    class_name: str
  ) -> "ClassSampler":
    """Create a sampler for one configured class.

    Args:
      class_group: Configuration group containing the class.
      class_name: Name of the class to sample.

    Returns:
      :class:`ClassSampler`: A sampler configured for the requested class.
    """

    # calls `self.resolve()`
    return ClassSampler(
      spec = self.resolve(class_group, class_name), 
      config_identity = self.identity,
    )

  def sample(
    self,
    class_group: str,
    class_name: str,
    rng: np.random.Generator,
    seed: int | None = None,
  ) -> SampledParameters:
    """Sample concrete parameters for one class instance.

    Args:
      class_group: Configuration group containing the class.
      class_name: Name of the class to sample.
      rng: Random-number generator used for stochastic sampling.
      seed: Optional per-sample seed recorded in the result or used to derive a reproducible sample.

    Returns:
      Concrete sampled parameters for one class instance.
    """

    # calls `ClassSampler.sample()` after `self.sampler()` creates one.
    return self.sampler(
      class_group = class_group, 
      class_name = class_name
    ).sample(
      rng = rng, 
      seed = seed
    )



@dataclass(frozen=True)
class ClassSampler:
  """Samples concrete parameters for one configured symbol class.

  A ``ClassSampler`` binds a resolved :class:`ClassSpec` to the identity of the configuration from which it was created. Each call to :meth:`sample` produces one independently sampled parameter set and records its sampling provenance.

  Attributes:
    spec: Resolved specification for the class being sampled.
    config_identity: Version and provenance information for the configuration used to construct this sampler.
  """

  spec: ClassSpec
  config_identity: ConfigIdentity

  def sample(
    self, 
    rng: np.random.Generator,
    seed: int | None = None,
  ) -> SampledParameters:
    """Sample one concrete instance of the configured class.

    The supplied random-number generator controls stochastic decisions.
    The optional seed is recorded in the resulting provenance metadata and may be used by callers to identify or reproduce the sample. It does not replace ``rng`` as the source of randomness.

    Args:
      rng: Random-number generator used for categorical, continuous and conditional sampling. 
      seed: Optional per-sample seed to record in provenance metadata.

    Returns:
      Sampled parameters, derived values, topology, component specifications, and provenance for one sampled class instance.

    Raises:
      SamplingValidationError: If the class specification is invalid or cannot be sampled.
      ValueError: If a sampled distribution produces invalid parameters.
    """

    # from `resolver.py`
    result = sample_class(
      spec = self.spec, 
      rng = rng
    )

    # from `provenance.py`
    provenance = create_sampling_provenance(
      config = self.config_identity,
      class_group = self.spec.class_group,
      class_name = self.spec.class_name,
      seed = seed,
      decisions = result.decisions,
      parameters = result.parameters,
      derived = result.derived,
    )

    return SampledParameters(
      parameters=result.parameters,
      derived=result.derived,
      topology=result.topology,
      components=result.components,
      provenance=provenance,
    )
