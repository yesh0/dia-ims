import argparse
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

    def set(self, value: typing.Any, *path: str):
        v = self.config
        for parent in path[:-1]:
            if parent not in v:
                v[parent] = {}
            v = v[parent]
        v[path[-1]] = value

    def optional(self, typer: typing.Type[T], default: T, *path: str) -> T:
        try:
            return self.require(typer, *path)
        except KeyError:
            return default


def get_config(args):
    parser = argparse.ArgumentParser(description="Processes IMS data files (.mzML)")
    parser.add_argument("files", metavar="mzML", nargs="+", type=str)
    parser.add_argument("-c", "--config", metavar="config", dest="config", type=argparse.FileType("rb"), required=True)
    parser.add_argument("-l", "--library", metavar=".FASTA", dest="library", type=str, required=False)
    args = parser.parse_args(args)
    config = Config(args.config)
    if args.library:
        config.set(args.library, "peptide_searcher", "library", "library_file")
    return config, args
