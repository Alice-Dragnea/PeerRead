# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="maxidl/Llama-OpenReviewer-8B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("maxidl/Llama-OpenReviewer-8B")
model = AutoModelForCausalLM.from_pretrained("maxidl/Llama-OpenReviewer-8B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = ${processorVarName}.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(${processorVarName}.decode(outputs[0][inputs["input_ids"].shape[-1]:]))