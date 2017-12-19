import json

import fachung.logger as logger

def read_experiment_configuration(config_path):
    with open(config_path, 'r') as config_file:
        data = (
            config_file
            .read()
            .replace('\n', '')
            .replace(' ', '')
            .strip()
        )
        configuration = json.loads(data)

    logger.INFO("Using Configuration %s" % configuration)
    return configuration
