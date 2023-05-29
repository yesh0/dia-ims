import logging
import os.path

import dia.config
from dia import plotting
from dia.workflow import DiaImsWorkflow

_logger = logging.getLogger(__package__)
_error = _logger.error


def main():
    logging.basicConfig(level=logging.INFO)
    config, args = dia.config.get_config(None)
    files: list[str] = [os.path.realpath(f) for f in args.files]
    for f in files:
        if not os.path.exists(f):
            _error("nonexistent data file: %s", f)
    plotting.set_global_style()
    for f in files:
        DiaImsWorkflow(config).process(f)


if __name__ == "__main__":
    main()
