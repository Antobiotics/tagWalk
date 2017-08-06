from __future__ import absolute_import

import click

import fachung.logger as l

from fachung.preparation.fashionista import Fashionista
from fachung.commands.cli import pass_context


@click.command('fashionista', short_help="Prepares Fashionista Data")
@click.option('--df/--not-df', default=False)
@click.option('--labels/--not-labels', default=False)
@click.option('--images/--not-images', default=False)
@pass_context
def cli(ctx, df, labels, images):

    l.INFO("Preparing ASOS data")
    prep = Fashionista()
    prep.prepare(df=df,
                 labels=labels,
                 images=images)



