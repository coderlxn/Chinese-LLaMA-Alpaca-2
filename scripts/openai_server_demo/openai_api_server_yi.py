import argparse
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread
from sse_starlette.sse import EventSourceResponse

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from peft import PeftModel

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
)

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"


def generate_prompt(
    instruction, response="", with_system_prompt=True, system_prompt=None
):
    if with_system_prompt is True:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map(
            {"instruction": instruction, "system_prompt": system_prompt}
        )
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({"instruction": instruction})
    if len(response) > 0:
        prompt += " " + response
    return prompt


def generate_completion_prompt(instruction: str):
    """Generate prompt for completion"""
    return generate_prompt(instruction, response="", with_system_prompt=True)


def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion"""

    system_msg = None
    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            if is_first_user_content is True:
                prompt += generate_prompt(
                    msg.content, with_system_prompt=True, system_prompt=system_msg
                )
                is_first_user_content = False
            else:
                prompt += "<s>" + generate_prompt(msg.content, with_system_prompt=False)
        if msg.role == "assistant":
            prompt += f" {msg.content}" + "</s>"
    return prompt


def predict(
    input,
    max_new_tokens=128,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    num_beams=1,
    repetition_penalty=1.1,
    do_sample=True,
    **kwargs,
):
    """
    Main inference method
    type(input) == str -> /v1/completions
    type(input) == list -> /v1/chat/completions
    """
    if isinstance(input, str):
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = False
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = float(repetition_penalty)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("[/INST]")[-1].strip()
    return output


def stream_predict(
    input,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    model_id="chinese-llama-alpaca-2",
    **kwargs,
):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    if isinstance(input, str):
        prompt = generate_completion_prompt(input)
    else:
        prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        streamer=streamer,
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=float(repetition_penalty),
    )
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    for new_text in streamer:
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield "[DONE]"


def get_embedding(input):
    """Get embedding main function"""
    with torch.no_grad():
        encoding = tokenizer(input, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        model_output = model(input_ids, attention_mask, output_hidden_states=True)
        data = model_output.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
        masked_embeddings = data * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_length = torch.sum(mask, dim=1)
        embedding = sum_embeddings / seq_length
        normalized_embeddings = F.normalize(embedding, p=2, dim=1)
        ret = normalized_embeddings.squeeze(0).tolist()
    return ret


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    msgs = request.messages
    if isinstance(msgs, str):
        msgs = [ChatMessage(role="user", content=msgs)]
    else:
        msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
    if request.stream:
        generate = stream_predict(
            input=msgs,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return EventSourceResponse(generate, media_type="text/event-stream")
    output = predict(
        input=msgs,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    # choices = [
    #     ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
    # ]
    choices = [
        ChatCompletionResponseChoice(
            index=0, message=ChatMessage(role="assistant", content=output)
        )
    ]
    return ChatCompletionResponse(choices=choices)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion"""
    output = predict(
        input=request.prompt,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [CompletionResponseChoice(index=0, text=output)]
    return CompletionResponse(choices=choices)


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Creates text embedding"""
    embedding = get_embedding(request.input)
    data = [{"object": "embedding", "embedding": embedding, "index": 0}]
    return EmbeddingsResponse(data=data)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="01-ai/Yi-6B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=True,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8111, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=6006, workers=1, log_config=log_config)
