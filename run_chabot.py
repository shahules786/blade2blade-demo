import os
from chatbot import generate_chat
from blade2blade import Blade2Blade
from transformers import Conversation

MODEL = "OpenAssistant/oasst-sft-1-pythia-12b"


safechat = int(os.environ.get("SAFE_CHAT",0))
if safechat:
    safetymodel = Blade2Blade("shahules786/blade2blade-t5-base")
else:
    safetymodel = None

def make_conversation(user_input,conv=None):
    if conv is None:
        conv = Conversation(user_input)
        return conv
    conv.add_user_input(user_input)
    return conv


def prepare_prebias(response):
    response = response.split("<sep>")
    label,rots = response[0],"and".join(response[1:]).strip("</s>")
    if label.strip() == "__casual__":
        return ""
    else:
        return f"Answer the following request as responsible chatbot that belives that {rots} : "

if __name__ == "__main__":

    conv = None
    exit = False
    while not exit:

        user_input = input("USER:")
        exit = True if user_input.lower() == "exit" else False
        if safechat:
            response, conv = safetymodel(user_input, conv)
            response = prepare_prebias(response)
        else:
            response = ""
            conv = make_conversation(user_input,conv)
        response = generate_chat(MODEL,
                      conv,
                      preprompt=response)
        conv.mark_processed()
        conv.append_response(response)
        print("OA:", response)

    
    

    

    