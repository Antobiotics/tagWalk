from __future__ import absolute_import

import click

import fachung.logger as logger
import fachung.configuration as configuration

from fachung.commands.cli import pass_context


@click.command('tagwalk', short_help="TagWalk Models")
@click.option('--push/--not-push', default=True)
@pass_context
def cli(ctx, push):
    logger.INFO("Managing Project data")

    if push:
        data_dir = configuration.BASE_DATA
        logger.INFO("Using: %s" % (data_dir))

