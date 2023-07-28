from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    # use_auth_token=True
)

prompt = "Give me a list of the Named Entitites (PERSON, LOCATION, ORGANIZATION, TIME, WORK_OF_ART, OCCUPATION) in the following sentence: 'Yumi Matsutoya (松任谷 由実, Matsutōya Yumi, born January 19, 1954), nicknamed Yuming (ユーミン, Yūmin), is a Japanese singer, composer, lyricist and pianist.'\n"

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