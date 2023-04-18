import typing

import tomli


T = typing.TypeVar("T")


class Config:
    """Reads config from a TOML file."""

    def __init__(self, f: typing.BinaryIO):
        self.config = tomli.load(f)

    def require(self, typer: typing.Type[T], *path: str) -> T:
        v = self.config
        try:
            for segment in path:
                v = v[segment]
            if isinstance(v, typer):
                return v
            else:
                return typer(v)
        except KeyError:
            raise KeyError("config item not found: %s", ".".join(path))

    def optional(self, typer: typing.Type[T], default: T, *path: str) -> T:
        try:
            return self.require(typer, *path)
        except KeyError:
            return default
