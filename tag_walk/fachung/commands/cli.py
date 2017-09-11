import os
import sys

import click

class Context(object):

    def __init__(self):
        self.verbose = False
        self.home = os.getcwd()

    def log(self, msg, *args):
        """Logs a message to stderr."""
        if args:
            msg %= args
        click.echo(msg, file=sys.stderr)

    def vlog(self, msg, *args):
        """Logs a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(msg, *args)


pass_context = click.make_pass_decorator(Context, ensure=True)


COMMANDS_FOLDER = os.path.join(os.path.dirname(__file__), '.')


class FachungCommand(click.MultiCommand):
    @property
    def sub_folder(self):
        raise RuntimeError("Missing Property: sub_folder")

    @property
    def base_folder(self):
        return '/'.join([COMMANDS_FOLDER, self.sub_folder])

    def is_command(self, filename):
        is_py = filename.endswith('.py')
        is_init = filename == '__init__.py'
        is_errors = filename == 'errors.py'
        return (
            is_py and
            not is_init and
            not is_errors
        )

    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(self.base_folder):
            if self.is_command(filename):
                rv.append(filename[:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        module_str = '.'.join(['fachung', 'commands', self.sub_folder])
        try:
            if sys.version_info[0] == 2:
                name = name.encode('ascii', 'replace')
            mod = __import__(module_str + '.' + name,
                             None, None, ['cli'])
        except ImportError as e:
            raise RuntimeError("Could not import commands, reason: %s" % (e))
        return mod.cli


class BuilderCommand(FachungCommand):
    @property
    def sub_folder(self):
        return 'builders'


class ModelingCommand(FachungCommand):
    @property
    def sub_folder(self):
        return 'modeling'
