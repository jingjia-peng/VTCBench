"""
Script to generate images from JSONL/JSON files with _context and _render_args fields.

This script:
1. Loads data from JSONL/JSON files (or optionally from HuggingFace datasets)
2. Generates images using the _context and _render_args from each entry
3. Saves results to JSONL format with updated image paths

Usage:
    # Basic usage - generate images from a JSONL file
    python examples/text2image_pipeline.py \
        --input_file data/my_dataset.jsonl \
        --output_dir data/my_dataset_generated \
        --render.saveImage True

    # Generate from a JSON file (array of objects)
    python examples/text2image_pipeline.py \
        --input_file data/my_dataset.json \
        --output_dir data/my_dataset_generated \
        --render.saveImage True \
        --num_workers 4

    # Test with a small number of examples
    python examples/text2image_pipeline.py \
        --input_file data/my_dataset.jsonl \
        --output_dir data/my_dataset_generated \
        --render.saveImage True \
        --num_examples 10

    # Use HuggingFace dataset (backward compatibility)
    python examples/text2image_pipeline.py \
        --data.path MLLM-CL/VTCBench \
        --data.split Retrieval \
        --output_dir data/VTCBench_regenerated \
        --render.saveImage True
"""

import asyncio
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

from deocr.engine.args import RenderArgs
from deocr.engine.playwright.async_api import transform
from jsonargparse import ArgumentParser
from tqdm import tqdm


def trim_img_path(img_path: str) -> str:
    """Trim image path to relative path from images/ directory."""
    out = img_path.split("images/", 1)[1]
    return f"images/{out}"


async def process_example_async(
    example: dict[str, Any],
    output_dir: str,
    render_args: RenderArgs,
) -> dict[str, Any]:
    """
    Process a single example asynchronously.

    Args:
        example: Example dictionary with _context and _render_args fields
        output_dir: Output directory for images
        render_args: Base rendering arguments

    Returns:
        Updated example dict
    """
    # Get the context text
    context = example.get("_context")
    if context is None:
        raise ValueError(f"Example missing '_context' field: {example.keys()}")

    # Parse _render_args if it's a string, otherwise use as-is
    example_render_args = example.get("_render_args")
    if example_render_args is None:
        raise ValueError(f"Example missing '_render_args' field: {example.keys()}")

    # If _render_args is a string, parse it as JSON or Python dict string
    if isinstance(example_render_args, str):
        try:
            example_render_args = json.loads(example_render_args)
        except json.JSONDecodeError:
            import ast
            try:
                example_render_args = ast.literal_eval(example_render_args)
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"Could not parse _render_args. Expected JSON or Python dict, "
                    f"got: {example_render_args[:100]}..."
                )

    # Merge example's render args with provided render args
    merged_render_args_dict = {
        **vars(render_args),
        **example_render_args,
    }
    # Ensure saveImage is True to save images
    merged_render_args_dict["saveImage"] = True

    merged_render_args = RenderArgs(**merged_render_args_dict)

    # Create images directory
    images_dir = osp.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Transform text to images
    image_paths = await transform(
        item=context,
        cache_dir=images_dir,
        render_args=merged_render_args,
    )

    # Update example with new image paths
    updated_example = example.copy()
    updated_example["images"] = [trim_img_path(p) for p in image_paths]

    return updated_example


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """
    Load data from a JSONL file (one JSON object per line).
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries
    """
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_json(file_path: str) -> list[dict[str, Any]]:
    """
    Load data from a JSON file (array of objects or single object).
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        List of dictionaries
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # If it's a single object, wrap it in a list
    if isinstance(data, dict):
        return [data]
    # If it's already a list, return it
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"JSON file must contain an object or array of objects, got {type(data)}")


def load_from_file(input_file: str) -> list[dict[str, Any]]:
    """
    Load data from a JSONL or JSON file.
    
    Args:
        input_file: Path to JSONL or JSON file
    
    Returns:
        List of dictionaries
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if input_file.endswith(".jsonl"):
        return load_jsonl(input_file)
    elif input_file.endswith(".json"):
        return load_json(input_file)
    else:
        # Try to auto-detect format
        try:
            return load_jsonl(input_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                return load_json(input_file)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Could not parse {input_file} as JSONL or JSON. "
                    f"Please ensure the file has .jsonl or .json extension, or is valid JSONL/JSON format."
                ) from e


def generate_images(
    input_file: str | None = None,
    data_path: str | None = None,
    split: str | None = None,
    output_dir: str | None = None,
    output_file: str | None = None,
    render_args: RenderArgs | None = None,
    num_workers: int = 1,
    num_examples: int | None = None,
):
    """
    Generate images for all examples in the input file or dataset.
    
    Args:
        input_file: Path to JSONL or JSON file with _context and _render_args fields
        data_path: (Optional) Path to HuggingFace dataset (for backward compatibility)
        split: (Optional) Dataset split for HuggingFace datasets
        output_dir: Directory to save generated JSONL and images
        output_file: (Optional) Specific output file path (default: input_file name in output_dir)
        render_args: Base rendering arguments
        num_workers: Number of parallel workers
        num_examples: Limit number of examples to process (for testing)
    """
    # Determine output directory and file
    if output_dir is None:
        raise ValueError("output_dir is required")
    
    if input_file is not None:
        # Load from JSONL/JSON file
        print(f"Loading data from {input_file}")
        examples = load_from_file(input_file)
        
        # Determine output filename
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.stem + ".jsonl"
        
        output_jsonl = osp.join(output_dir, output_file)
    elif data_path is not None and split is not None:
        # Backward compatibility: Load from HuggingFace dataset
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFace dataset loading. "
                "Install it with: pip install datasets"
            )
        
        print(f"Loading dataset from {data_path}, split: {split}")
        dataset = load_dataset(
            data_path,
            split=split,
            columns=["problem", "answers", "images", "_context", "_render_args"],
        )
        examples = list(dataset)
        output_jsonl = osp.join(output_dir, f"{split}.jsonl")
    else:
        raise ValueError(
            "Either input_file (for JSONL/JSON) or both data_path and split "
            "(for HuggingFace datasets) must be provided"
        )
    
    # Limit number of examples if specified
    if num_examples is not None:
        examples = examples[:num_examples]
    
    print(f"Processing {len(examples)} examples...")
    
    # Validate that examples have required fields
    for i, example in enumerate(examples[:5]):  # Check first 5 examples
        if "_context" not in example:
            raise ValueError(
                f"Example {i} is missing '_context' field. "
                f"Available fields: {list(example.keys())}"
            )
        if "_render_args" not in example:
            raise ValueError(
                f"Example {i} is missing '_render_args' field. "
                f"Available fields: {list(example.keys())}"
            )
    
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for processing
    if render_args is None:
        render_args = RenderArgs()
    render_args_dict = vars(render_args)
    tasks = [
        (example, output_dir, render_args_dict)
        for example in examples
    ]
    
    async def process_all_examples():
        """Process all examples concurrently in a single async function using asyncio.gather."""
        coros = []
        for example, output_dir, render_args_dict in tasks:
            render_args = RenderArgs(**render_args_dict)
            coros.append(process_example_async(example, output_dir, render_args))
        results = []
        for f in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Generating images"):
            result = await f
            results.append(result)
        return results
    
    # Run all processing in a single event loop
    results = asyncio.run(process_all_examples())
    
    # Write results to JSONL
    print(f"Writing results to {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Done! Generated images for {len(results)} examples.")
    print(f"Output directory: {output_dir}")
    print(f"JSONL file: {output_jsonl}")
    print(f"Images directory: {osp.join(output_dir, 'images')}")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Input options - either JSONL/JSON file or HuggingFace dataset
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to JSONL or JSON file with _context and _render_args fields",
    )
    input_group.add_argument(
        "--data.path",
        type=str,
        default=None,
        help="(Optional) HuggingFace dataset path for backward compatibility",
    )
    input_group.add_argument(
        "--data.split",
        type=str,
        default=None,
        choices=["Retrieval", "Reasoning", "Memory"],
        help="(Optional) Dataset split for HuggingFace datasets",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated JSONL and images",
    )
    output_group.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="(Optional) Output filename (default: input filename with .jsonl extension)",
    )
    
    # Processing options
    processing_group = parser.add_argument_group("Processing options")
    processing_group.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    processing_group.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )
    
    parser.add_class_arguments(RenderArgs, "render")
    
    args = parser.parse_args()
    
    generate_images(
        input_file=args.input_file,
        data_path=args.data.path,
        split=args.data.split,
        output_dir=args.output_dir,
        output_file=args.output_file,
        render_args=args.render,
        num_workers=args.num_workers,
        num_examples=args.num_examples,
    )
