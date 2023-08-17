from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    # use_auth_token=True
)

#prompt = "Give me a list of the Named Entitites (PERSON, LOCATION, ORGANIZATION, TIME, WORK_OF_ART, OCCUPATION) in the following sentence: 'Yumi Matsutoya (松任谷 由実, Matsutōya Yumi, born January 19, 1954), nicknamed Yuming (ユーミン, Yūmin), is a Japanese singer, composer, lyricist and pianist.'\n"

prompt = """
[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.\n\n
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\nHi, how are you? [/INST] Good thanks! \n[INST] What is the capital of France? [/INST]
"""

sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")