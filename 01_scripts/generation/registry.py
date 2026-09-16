from .primitive.ch import generate_ch
from .primitive.slst import generate_slst
from .primitive.sc import generate_sc
from .primitive.hdc import generate_hdc
from .primitive.dc import generate_dc
from .primitive.tr import generate_tr
from .primitive.dtr import generate_dtr
from .compound.together import generate_together
from .compound.increase import generate_increase
from .compound.crossed import generate_crossed
from .compound.ch3picot import generate_ch3picot
from .instructive.ring import generate_ring
from .instructive.loop import generate_loop


GENERATOR_REGISTRY = {
  ("primitive", "ch"): generate_ch,
  ("primitive", "slst"): generate_slst,
  ("primitive", "sc"): generate_sc,
  ("primitive", "hdc"): generate_hdc,
  ("primitive", "dc"): generate_dc,
  ("primitive", "tr"): generate_tr,
  ("primitive", "dtr"): generate_dtr,
  ("compound", "together"): generate_together,
  ("compound", "increase"): generate_increase,
  ("compound", "crossed"): generate_crossed,
  ("compound", "ch3picot"): generate_ch3picot,
  ("instructive", "ring"): generate_ring,
  ("instructive", "loop"): generate_loop,
}

# These generators consume the explicit CompositeSample produced by
# SamplingConfig.realize_components(), rather than a bare parent sample.
COMPOSITE_GENERATORS = {
  ("compound", "together"),
  ("compound", "increase"),
  ("compound", "crossed"),
  ("compound", "ch3picot"),
  ("instructive", "ring"),
}

EXPECTED_PRIMITIVE_GENERATORS = {
  "ch", "slst", "sc", "hdc", "dc", "tr", "dtr",
}

CLASS_GROUPS = {
  "primitive": [
    "ch", 
    "slst",
    "sc",
    "hdc",
    "dc",
    "tr",
    "dtr",
  ],
  "compound": [
    "together",
    "increase",
    "post",
    "crossed",
    "rounded",
    "ch3picot",
  ],
  "instructive": [
    "ring", 
    "loop",
    "jb",
    "arrow",
  ]
}
