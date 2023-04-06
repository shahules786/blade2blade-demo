from regex import R
from text_generation import InferenceAPIClient
import os
from transformers import Conversation

import logging
logger = logging.getLogger('spam_application')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

HF_TOKEN = os.environ.get("HF_TOKEN")
SPECIAL_TOKENS = {"user": "<|prompter|>",
                   "response": "<|system|>",
                   "eos":"<|endoftext|>"}

def prepare_conversation(conv):
    
    
    conversation = ["{}{}{}".format(SPECIAL_TOKENS["user"] if is_user else SPECIAL_TOKENS["response"],text,SPECIAL_TOKENS["eos"])
                    for is_user,text in conv.iter_texts()]
    
    return "".join(conversation)

def prepare_input(input_text,preprompt):
    
    if preprompt=="":
        return input_text
    input_list = input_text.split(SPECIAL_TOKENS["user"])
    input_list[-1] = preprompt + input_list[-1]
    return SPECIAL_TOKENS["user"].join(input_list)

def generate_chat(
    model: str,
    conversation: Conversation,
    prebias: str = "",
    preprompt: str = "",
    typical_p: float = 0.2,
    top_p: float = 0.25,
    temperature: float = 1.5,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    truncate: int = 1000,
    watermark: bool = False,
    max_new_tokens: int = 550,
):

    client = InferenceAPIClient(model, token=HF_TOKEN)
    input_text = prepare_conversation(conversation)
    input_text = prepare_input(input_text,preprompt)

    logger.debug(input_text)
    iterator = client.generate_stream(
        input_text,
        top_p=top_p,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        typical_p=typical_p,
        truncate=truncate,
        watermark=watermark,
        max_new_tokens=max_new_tokens,
    )

    response = "".join([item.token.text for item in iterator if not item.token.special])
    return response



