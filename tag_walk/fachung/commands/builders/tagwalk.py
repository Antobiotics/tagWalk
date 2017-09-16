from __future__ import absolute_import

import click

import fachung.logger as logger

from fachung.preparation.tagwalk import TagWalk
from fachung.commands.cli import pass_context


@click.command('tagwalk', short_help="Prepares TagWalk Data")
@click.option('--df/--not-df', default=False)
@click.option('--images/--not-images', default=False)
@pass_context
def cli(ctx, df, images):

    logger.INFO("Preparing TagWalk data")
    prep = TagWalk()
    prep.prepare(df=df,
                 images=images)
