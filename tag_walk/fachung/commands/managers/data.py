from __future__ import absolute_import

import click
from execute import execute

import fachung.logger as logger
import fachung.configuration as configuration

from fachung.commands.cli import pass_context

def _execute(command):
    logger.INFO(command)
    return execute(command)

def package_data(archive, data_dir):
    command = """
    tar -zcvf {archine} {data_dir}
    """.format(archive=archive,
               data_dir=data_dir)

    res = _execute(command)
    if res is  None:
        raise RuntimeError("Unable to package data")

def push_data():
    pass


@click.command('tagwalk', short_help="TagWalk Models")
@click.option('--package/--not-package', default=False)
@click.option('--push/--not-push', default=False)
@click.option('--archive_name', default='data.tar.gz')
@pass_context
def cli(ctx, package, push, archive_name):
    logger.INFO("Managing Project data")

    data_dir = configuration.BASE_DATA
    logger.INFO("Using: %s" % (data_dir))

    if package:
        package_data(archive_name, data_dir)

    if push:
        push_data()
