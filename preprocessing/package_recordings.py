from preprocessing import initialize_preprocessing
from utils.logger import get_logger
import argparse
from .components.participants import construct_participants_table


def main(args):
    config = initialize_preprocessing(args.config)
    logger = get_logger()
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content: \n{config}")

    construct_participants_table(
        config, participant_id=args.participant, session=args.session
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Package recordings based on a configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--participant", type=str, default=None, help="Participant ID to process."
    )
    parser.add_argument(
        "--session", type=str, default=None, help="Session ID to process."
    )
    args = parser.parse_args()

    main(args)
