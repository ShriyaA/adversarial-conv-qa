import argparse
import wandb
import logging
import utils
from utils import config_utils
from utils import logging_utils

from agents import *


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = config_utils.process_config(args.config)

    if config.wandb:
        #setup wandb
        wandb.init()
        run_name = wandb.run.name
    else: 
        run_name = utils.misc.get_random_string(8)
        print("Locally generated run_id is ", run_name)


    config_utils.create_config_dirs(config, run_name)

    if config.wandb:
        for key,value in wandb.config.items(): # If values are passed from sweep, overwrite them in config
            if key in config:
                config[key] = value
    
    # setup the logger
    logging_utils.setup_logging(config.log_dir)

    # log config
    logger = logging.getLogger()
    logger.info(config)
    config_utils.print_config(config)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()