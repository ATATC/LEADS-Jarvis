from argparse import ArgumentParser as _ArgumentParser
from sys import exit as _exit

from leads import register_config as _register_config, load_config as _load_config
from leads_gui import Config as _Config

from leads_jarvis.cli import main


def __entry__() -> None:
    parser = _ArgumentParser(prog="LEADS Jarvis",
                             description="Jarvis Extension for LEADS",
                             epilog="GitHub: https://github.com/ProjectNeura/LEADS-Jarvis")
    parser.add_argument("-c", "--config", default=None, help="specify a configuration file")
    args = parser.parse_args()
    _register_config(_load_config(args.config, _Config) if args.config else _Config({}))
    _exit(main())
