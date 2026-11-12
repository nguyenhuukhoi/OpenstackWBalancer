"""Console entry points.

The CLI modules import openstacksdk at load time. openstacksdk is installed by
the operator (see compute_maintenance.dependencies), so check it first and turn
a bare ModuleNotFoundError traceback into install instructions.
"""
import sys

import compute_maintenance.config as _config
import compute_maintenance.dependencies as _deps


def require_sdk(error_exit, stderr=None):
    """Exit with ``error_exit`` when openstacksdk is missing; print warnings."""
    stderr = stderr or sys.stderr
    errors, warnings = _deps.check_sdk()
    for warning in warnings:
        print("WARNING: " + warning, file=stderr)
    if errors:
        for error in errors:
            print("ERROR: " + error, file=stderr)
        print(_deps.install_hint(), file=stderr)
        raise SystemExit(error_exit)


def balance():
    # Icinga treats 3 as UNKNOWN; any other failure of an apply run is 1.
    require_sdk(_config.ICINGA_UNKNOWN if "--monitor-only" in sys.argv[1:] else 1)
    from compute_maintenance.cli.balance import main
    main()


def maintain(argv=None):
    require_sdk(1)
    from compute_maintenance.cli.maintain import main
    raise SystemExit(main(argv))
