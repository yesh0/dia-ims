import argparse
import logging
import os.path
import sys

from dia import plotting
from dia.config import Config
from dia.workflow import DiaImsWorkflow


_logger = logging.getLogger(__package__)
_error = _logger.error


def main():
    logging.basicConfig(stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Processes IMS data files (.mzML)")
    parser.add_argument("files", metavar="mzML", nargs="+", type=str)
    parser.add_argument("-c", "--config", metavar="config", dest="config", type=argparse.FileType("rb"), required=True)
    args = parser.parse_args()
    config = Config(args.config)
    files: list[str] = [os.path.realpath(f) for f in args.files]
    for f in files:
        if not os.path.exists(f):
            _error("nonexistent data file: %s", f)
    plotting.set_global_style()
    for f in files:
        DiaImsWorkflow(config).process(f)


if __name__ == "__main__":
    main()
