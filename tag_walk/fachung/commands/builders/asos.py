from __future__ import absolute_import

import click

import fachung.logger as logger

from fachung.preparation.asos import Asos
from fachung.commands.cli import pass_context


@click.command('asos', short_help="Prepares Asos Data")
@click.option('--reset/--not-rest', default=False)
@click.option('--df/--not-df', default=False)
@click.option('--labels/--not-labels', default=False)
@click.option('--images/--not-images', default=False)
@pass_context
def cli(ctx, reset, df, labels, images):

    logger.INFO("Preparing ASOS data")
    prep = Asos(build=reset)
    prep.prepare(df=df,
                 labels=labels,
                 images=images,
                 reset=reset)
