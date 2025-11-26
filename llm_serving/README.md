# vLLM Serving

This project expects that LLM/VLM is seperately deployed, such as [vLLM](https://github.com/vllm-project/vllm), OpenAI API, etc.

A simple example to get you started, using deps from [pyproject.toml](./pyproject.toml):

```sh
# set up a vllm environment seperately, using pyproject.toml
uv venv
uv sync --all-extras
# serve your model
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8001
# to test your endpoint
curl http://localhost:8001/v1/models
```

## Known Dependency Constraints

| Model Name | Dependency |
|:----------:|:----------:|
| [Qwen3-VL Series](https://huggingface.co/collections/Qwen/qwen3-vl)  | `vllm==0.11.0, transformers==4.57.1` |
| [moonshotai/Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) | `vllm==0.9.2, transformers<4.54` |
| [InternVL3.5 Series](https://huggingface.co/collections/OpenGVLab/internvl35) | `vllm==0.10.1.1, transformers==4.57.1` |
| [Phi-4-multimodal-instruct] | `vllm==0.8.2,transformers==4.48.2` |
