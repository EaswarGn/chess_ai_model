import sys

from trainer import HarmoniaTrainer
import argparse
from configs import import_config


def main(config):
    
    trainer = HarmoniaTrainer(config)

    """
    model = initialize_model(tokenizer, **config.model_config)

    trainer.setup_dataloaders(
        tokenizer,
        train_dataset,
        val_dataset,
        test_dataset,
    )
    trainer.setup_model(model)

    trainer.train()
    trainer.test()

    trainer.cleanup()"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    config = import_config(args.config_name)
    config = config.config()
    main(config)