from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from ..models import ConfigIdentity, SampledParameters
from .resolver import resolve_class_spec
from .schema import ClassSpec
from .sampler import ClassSampler

ClassKey = tuple[str, str]
CacheMode = Literal["lazy", "eager"]

@dataclass
class SamplingConfig:
  """Validated sampling system with cached specs and class samplers.

  ``SamplingConfig`` is the runtime entry point for a validated symbol-sampling configuration. It stores the sampling configuration and its identity, resolves class specifications on demand, and caches reusable class-specific samplers. Class specifications and class-specific samplers are resolved lazily by default, or eagerly during initialization when ``cach_mode`` is ``"eager"``.

  Use :meth:`sampler` when repeatedly sampling instances off one class. Use :meth:`sample` as a convenience method for one-off sampling requests.

  The public ``specs`` and ``samplers`` properties expose read-only mapping views over their respective internal caches. In ``"lazy"`` mode, these views may contain only entries that have already been resolved or prepared.

  Attributes:
    ontology_config: Parsed and validated ontology configuration mapping.
    sampling_config: Parsed and validated sampling configuration mapping.
    identity: Version and provenance information for the configuration.

    cache_mode: Cache initialization strategy. ``"lazy"`` resolves all classes when requested; ``"eager"`` prepares all configured classes during initialization.
    
    _spec_cache: Internal mutable cache of resolved class specifications keyed by ``(class_group, class_name)``.
    _spec_view: Lazily created read-only mapping view over ``_spec_cache``.
    
    _sampler_cache: Internal mutable cache of reusable class samplers keyed by ``(class_group, class_name)``.
    _sampler_view: Lazily created read-only mapping view over ``_sampler_cache``.

  Raises:
    ValueError: If ``cache_mode`` is not ``"lazy"`` or ``"eager"``.  
  """

  ontology_config: Mapping[str, Any]
  sampling_config: Mapping[str, Any]
  identity: ConfigIdentity

  cache_mode: CacheMode = "lazy"

  _spec_cache: dict[ClassKey, ClassSpec] = field(default_factory=dict, init=False, repr=False)
  _spec_view: Mapping[ClassKey, ClassSpec] | None = field(default=None, init=False, repr=False)

  _sampler_cache: dict[ClassKey, ClassSampler] = field(default_factory=dict, init=False, repr=False)
  _sampler_view: Mapping[ClassKey, ClassSampler] | None = field(default=None, init=False, repr=False)

  # `dataclass` automatically calls this after `__init__`
  def __post_init__(self) -> None:
    """Validates ``cache_mode``, then optionally prepares all class samplers.
    
    Raises: 
      ValueError: If ``cache_mode`` is not ``"lazy"`` or ``"eager"``.
    """

    if self.cache_mode not in ("lazy", "eager"):
      raise ValueError(f"Unsupported cache mode: {self.cache_mode!r}")
    
    # if eager, prepare all in class samplers immediately
    if self.cache_mode == "eager":
      self.prepare_all()

  @property
  def class_keys(self) -> tuple[ClassKey, ...]:
    """Returns all ``(class_group, class_name)`` pairs in canonical ontology order.

    The ontology defines the canonical ordering. Each key is represented as ``(class_group, class_name)``, where ``class_group`` corresponds to the ontology family and ``class_name`` is the class identifer.

    Returns:
      A tuple of configured class keys in ontology order.
    """

    return tuple(
      (class_entry["family"], class_entry["name"])
      for class_entry in self.ontology_config["classes"]
    )

  @property
  def specs(self) -> Mapping[ClassKey, ClassSpec]:
    """Return the read-only view of the resolved-specification cache.
    
    In lazy mode, this mapping may contain only specifications that have already been resolved. Accessing this property does not resolve additional classes.
    """

    if self._spec_view is None:
      self._spec_view = MappingProxyType(self._spec_cache)
    return self._spec_view

  @property
  def samplers(self) -> Mapping[ClassKey, ClassSampler]:
    """Return the read-only view of the class-sampler cache.
    
    In lazy mode, this mapping may contain only samplers that have already been requested. Accessing this property does not prepare additional samplers.

    Use :meth:`prepare_all` to populate the cache for every configured class.
    """

    if self._sampler_view is None:
      self._sampler_view = MappingProxyType(self._sampler_cache)
    return self._sampler_view


  def resolve(
    self, 
    class_group: str, 
    class_name: str
  ) -> ClassSpec:
    """Resolve and cache one class specification :class:`ClassSpec`.

    Args:
      class_group: Configuration group containing the class.
      class_name: Name of the class to resolve.

    Returns:
      :class:`ClassSpec`: The cached or newly resolved class specification.

    Raises:
      KeyError: If the group or class does not exist.
      SamplingValidationError: If the class specification is invalid.    
    """

    key = (class_group, class_name)

    # cache check, otherwise make spec
    if key not in self._spec_cache:
      self._spec_cache[key] = resolve_class_spec(
        ontology_config = self.ontology_config,
        sampling_config = self.sampling_config, 
        class_group = class_group, 
        class_name = class_name,
      )

    return self._spec_cache[key]

  def sampler(
    self, 
    class_group: str, 
    class_name: str
  ) -> "ClassSampler":
    """Return the cached sampler :class:`ClassSampler` for one class.

    Args:
      class_group: Configuration group containing the class.
      class_name: Name of the class to resolve.

    Returns:
      :class:`ClassSampler`: A reusable sampler bound to the resolved class specification.    
    """

    key = (class_group, class_name)

    # cache check, otherwise make sampler
    if key not in self._sampler_cache:
      self._sampler_cache[key] = ClassSampler(
        spec = self.resolve(*key), 
        config_identity = self.identity
      )

    # print("VARIANTS:")
    # for name, bundle in self._spec_cache[key].variants.items():
    #   print(name, bundle)
  
    # print("\nPARAMETERS:")
    # for name, parameter in self._spec_cache[key].parameters.items():
    #   print(name, parameter)

    return self._sampler_cache[key]

  def prepare_specs(self) -> None:
    """Eagerly resolve and cache every configured class specification."""
    for key in self.class_keys:
      self.resolve(*key)

  def prepare_all(self) -> None:
    """Eagerly resolve and cache every spec and class sampler."""
    self.prepare_specs()
    for key in self.class_keys:
      self.sampler(*key)

  def sample(
    self, 
    class_group: str, 
    class_name: str, 
    rng: np.random.Generator, 
    *, 
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    case_id: str | None = None,
  ) -> SampledParameters:
    """Sample one instance using the cached class sampler.

    This is a convenience method equivalent to calling :meth:`sampler` followed by :meth:`ClassSampler.sample`.
    """

    return self.sampler(class_group, class_name).sample(
      rng,
      seed=seed,
      overrides=overrides,
      case_id=case_id,
    )
