from luigi.configuration import LuigiConfigParser

def get_config_(path = '/opt/cctv/fachung.cfg'):
    config = LuigiConfigParser.instance()

    if not path is None:
        config.add_config_path(path)

    return config

CONFIGURATION = get_config_()

def get_config():
    return CONFIGURATION

MODE = (
    get_config()
    .get('project', 'mode')
)

HOME = (
    get_config()
    .get(MODE, 'base')
)

PROJECT = (
    get_config()
    .get(MODE, 'project')
)

DATA = (
    get_config()
    .get(MODE, 'data')
)

BASE_DATA = '/'.join([HOME, PROJECT, DATA])
