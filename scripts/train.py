import os, sys, hydra
sys.path.append('..')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from learnable_typewriter.utils.defaults import PROJECT_PATH
from learnable_typewriter.trainer import Trainer

if __name__ == "__main__":
    config_path, config_name = os.path.split(sys.argv[1])
    config_path = (config_path if len(config_path) else str(PROJECT_PATH/"configs"))

    @hydra.main(config_path=config_path, config_name=config_name)
    def run(config):
        seed = config['training'].get('seed', 4321)
        trainer = Trainer(config, seed=seed)
        trainer.run(seed=seed)

    sys.argv.pop(1)
    run()
