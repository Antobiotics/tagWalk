from __future__ import absolute_import

import click

import fachung.logger as logger
import fachung.experiment as exp

from fachung.commands.cli import pass_context
from fachung.models.tagwalk_cnn_rnn import TagWalkCNNRNN
from fachung.models.tagwalk_classifier import TagWalkClassifier


@click.command('tagwalk', short_help="TagWalk Models")
@click.option('--model', type=click.Choice(['cnn', 'cnn-rnn']))
@click.option('--config')
@click.option('--train/--not-train', default=True)
@pass_context
def cli(ctx, model, config, train):
    logger.INFO("Managing TagWalk Models")

    model_cls = TagWalkCNNRNN
    if model == 'cnn':
        model_cls = TagWalkClassifier

    logger.INFO("Using model: %s" % model_cls.__name__)

    configuration = exp.read_experiment_configuration(config)

    engine = model_cls(configuration)
    _ = engine.run(training=train)
