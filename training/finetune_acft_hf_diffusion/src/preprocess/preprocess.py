# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Read the args from preprocess component."""

import json
import argparse
from pathlib import Path
from argparse import Namespace


from azureml.acft.contrib.hf.diffusion.task_factory import get_task_runner
from azureml.acft.contrib.hf.diffusion.constants.constants import SaveFileConstants

from azureml.acft.accelerator.utils.logging_utils import get_logger_app
from azureml.acft.accelerator.utils.error_handling.exceptions import ValidationException
from azureml.acft.accelerator.utils.error_handling.error_definitions import PathNotFound
from azureml.acft.accelerator.utils.decorators import swallow_all_exceptions
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore


logger = get_logger_app()


def str2bool(arg):
    """Convert string to bool."""
    arg = arg.lower()
    if arg in ["true", '1']:
        return True
    elif arg in ["false", '0']:
        return False
    else:
        raise ValueError(f"Invalid argument {arg} to while converting string to boolean")


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model Preprocessing", allow_abbrev=False)

    parser.add_argument(
        "--output_dir",
        default="preprocess_output",
        type=str,
        help="folder to store preprocessed input data",
    )

    # Task settings
    parser.add_argument(
        "--model_selector_output",
        default=None,
        type=str,
        help=(
            "output folder of model selector containing model configs, tokenizer, checkpoints in case of model_id."
            "If huggingface_id is selected, the model download happens dynamically on the fly"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="StableDiffusion",
        help="Task Name",
    )

    return parser


def pre_process(parsed_args: Namespace, unparsed_args: list):
    """Pre process data."""
    Path(parsed_args.output_dir).mkdir(exist_ok=True)

    # Model Selector Component ---> Preprocessor Component
    model_selector_args_path = Path(parsed_args.model_selector_output, SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH)
    if not model_selector_args_path.exists():
        raise ValidationException._with_error(AzureMLError.create(PathNotFound, path=model_selector_args_path))

    with open(model_selector_args_path, "r") as rptr:
        model_selector_args = json.load(rptr)
        parsed_args.model_name = model_selector_args.get("model_name")
        model_name_or_path = Path(parsed_args.model_selector_output, parsed_args.model_name)
        if model_name_or_path.is_dir():
            parsed_args.model_name_or_path = model_name_or_path
        else:
            parsed_args.model_name_or_path = parsed_args.model_name

    # Preprocessing component has `unparsed args` which will be parsed and returned after this method
    hf_task_runner = get_task_runner(task_name=parsed_args.task_name)()
    hf_task_runner.run_preprocess_for_finetune(parsed_args, unparsed_args)  # type: ignore


@swallow_all_exceptions(logger)
def main():
    """Parse args and pre process."""
    parser = get_parser()
    # unknown args are the command line strings that could not be parsed by the argparser
    parsed_args, unparsed_args = parser.parse_known_args()
    logger.info(f"Component Args: {parsed_args}")

    pre_process(parsed_args, unparsed_args)


if __name__ == "__main__":
    main()
