from .primitive.sc import generate_sc


GENERATOR_REGISTRY = {
  "sc": generate_sc,
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
