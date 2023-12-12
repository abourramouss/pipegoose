import requests
from transformers import AutoConfig, AutoModelForCausalLM


def fetch_most_downloaded_text_generation_models(exclude_models=None):
    if exclude_models is None:
        exclude_models = []

    response = requests.get(
        "https://huggingface.co/api/models",
        params={
            "filter": "text-generation",
            "sort": "downloads",
            "direction": "-1",
            "limit": "20",
        },
    )

    if response.status_code != 200:
        raise Exception("Failed to fetch models from Hugging Face")

    model_ids = [
        model["modelId"]
        for model in response.json()
        if model["modelId"] not in exclude_models
    ]

    return model_ids


excluded_models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "davidkim205/komt-mistral-7b-v1",
    "tiiuae/falcon-40b-instruct",
    "petals-team/StableBeluga2",
    "TheBloke/CodeLlama-34B-Instruct-GPTQ",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
]

model_ids = fetch_most_downloaded_text_generation_models(exclude_models=excluded_models)

for model_id in model_ids:
    print(f"Model ID: {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config)
    print(model)
