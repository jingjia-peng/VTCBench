import json
from copy import deepcopy
from glob import iglob

from jsonargparse import ArgumentParser

# from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from locoxim.args import DataArgs, ModelArgs, RunArgs
from locoxim.async_evaluate import NeedleHaystackTester
from locoxim.dataio import NeedleTestConfig


def worker(kwargs):
    tester = NeedleHaystackTester(**kwargs)
    tester.evaluate()


def run_test(
    model_args: ModelArgs,
    data_args: DataArgs,
    run_args: RunArgs,
):
    with open(data_args.needle_set_path, "r") as file:
        _raw_dict: list[dict] = json.load(file)
        experiment_config: list[NeedleTestConfig] = [
            NeedleTestConfig(**e) for e in _raw_dict
        ]

    # an experiment is a json, containing multiple tests, with a test_id and its args
    questions = [
        question
        for test_config in experiment_config
        for question in test_config.iter_question_items(
            default_system_prompt=model_args.system_prompt,
            default_task_template=data_args.task_template,
            base_seed=run_args.base_seed,
        )
    ]

    tasks: list[dict] = []
    for haystack_path in iglob(f"{data_args.haystack_dir}/*"):
        # increment base seed for every different haystack
        for question_item in questions:
            tasks.append(
                {
                    "model_args": deepcopy(model_args),
                    "data_args": deepcopy(data_args),
                    "run_args": deepcopy(run_args),
                    "question_item": question_item,
                    "haystack_path": haystack_path,
                }
            )
        run_args.base_seed = run_args.base_seed + 100

    for task in tqdm(tasks):
        worker(task)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_class_arguments(ModelArgs, "model")
    parser.add_class_arguments(DataArgs, "data")
    parser.add_class_arguments(RunArgs, "exp")

    args = parser.parse_args()

    model_args: ModelArgs = args.model
    data_args: DataArgs = args.data
    run_args: RunArgs = args.exp

    run_test(
        model_args=model_args,
        data_args=data_args,
        run_args=run_args,
    )
