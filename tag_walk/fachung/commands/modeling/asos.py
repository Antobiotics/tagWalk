from __future__ import absolute_import

import click

import fachung.logger as logger
import fachung.experiment as exp

from fachung.commands.cli import pass_context
from fachung.models.asos_siamese import AsosSiameseTrainer


@click.command('tagwalk', short_help="TagWalk Models")
@click.option('--config')
@click.option('--train/--not-train', default=True)
@pass_context
def cli(ctx, config, train):
    logger.INFO("Managing TagWalk Models")
    logger.INFO("Using model: %s" % AsosSiameseTrainer.__name__)

    configuration = exp.read_experiment_configuration(config)

    engine = AsosSiameseTrainer(configuration)
    _ = engine.run(training=train)
