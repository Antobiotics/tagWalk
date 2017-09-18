from __future__ import absolute_import

from datetime import datetime

import click
from executor import execute

import fachung.logger as logger
import fachung.configuration as configuration

from fachung.commands.cli import pass_context

def _execute(command):
    logger.INFO(command)
    return execute(command)

def package_data(archive, data_dir):
    command = """
    tar -zcvf {archive} {data_dir}
    """.format(archive=archive,
               data_dir=data_dir)

    res = _execute(command)
    if res is None:
        raise RuntimeError("Unable to package data")

def push_archive(archive):
    command = """
    aws cp {date}__{archive} s3://fachung/archives/ --profile fachung
    """.format(date=str(datetime.now().date()),
               archive=archive)
    res = _execute(command)
    if res is None:
        raise RuntimeError("Unable to copy data")

def clean_archive(archive):
    command = "rm %s" % (archive)
    _ = _execute(command)

def sync_data(data_dir):
    command = """
    aws s3 sync {data_dir} s3://fachung/data/ --profile fachung
    """.format(data_dir=data_dir)
    res = _execute(command)
    if res is None:
        raise RuntimeError("Unable to sync data")


@click.command('tagwalk', short_help="TagWalk Models")
@click.option('--package/--not-package', default=False)
@click.option('--push/--not-push', default=False)
@click.option('--clean/--not-clean', default=False)
@click.option('--sync/--not-sync', default=False)
@click.option('--archive_name', default='data.tar.gz')
@click.option('--data_dir', default='./data')
@pass_context
def cli(ctx, package, push, clean, sync, archive_name, data_dir):
    logger.INFO("Managing Project data")

    data_dir = configuration.BASE_DATA
    logger.INFO("Using: %s" % (data_dir))

    if package:
        package_data(archive_name, data_dir)

    if push:
        push_archive(archive_name)

    if clean:
        clean_archive(archive_name)

    if sync:
        sync_data(data_dir)
