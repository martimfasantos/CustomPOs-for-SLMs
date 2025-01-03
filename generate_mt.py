import torch
from transformers import pipeline

model_id = "martimfasantos/EuroLLM-1.7B-Instruct-DPO-gamma"

generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
outputs = generator([{"role": "user", 
                      "content": "Translate the following English source text to German:\n \
                                  English: Police arrest 15 after violent protest outside UK refugee hotel \n \
                                  German: "
                    }], do_sample=False, max_new_tokens=200)
print(outputs[0]['generated_text'])
