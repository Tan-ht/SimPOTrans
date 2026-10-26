from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer

__all__ = ["setup_my_fairseq2_extension"]


def setup_my_fairseq2_extension(container: "DependencyContainer") -> None:
    from .recipe import register_st_components

    register_st_components(container)
