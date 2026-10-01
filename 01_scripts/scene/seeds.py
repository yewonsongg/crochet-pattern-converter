"""Stable random seeds that do not depend on iteration order."""

from __future__ import annotations

import hashlib


def derive_seed(scene_seed: int, *parts: object) -> int:
  payload = "\x1f".join([str(scene_seed), *(str(part) for part in parts)])
  digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
  return int.from_bytes(digest, "big", signed=False)
