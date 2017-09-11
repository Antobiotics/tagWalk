from __future__ import absolute_import

import json
import click

import fachung.logger as logger

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

    with open(config, 'r') as config_file:
        data = (
            config_file
            .read()
            .replace('\n', '')
            .replace(' ', '')
            .strip()
        )
        configuration = json.loads(data)

    logger.INFO("Using Configuration %s" % configuration)

    engine = model_cls(configuration)
    _ = engine.run(training=train)
