from .primitive.ch import generate_ch
from .primitive.slst import generate_slst
from .primitive.sc import generate_sc


GENERATOR_REGISTRY = {
  ("primitive", "ch"): generate_ch,
  ("primitive", "slst"): generate_slst,
  ("primitive", "sc"): generate_sc,
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
