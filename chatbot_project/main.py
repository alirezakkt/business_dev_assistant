from transformers import AutoModelForCausalLM, AutoTokenizer

# loading model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt" ,padding = True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(input_ids, attention_mask = attention_mask, max_length=150, num_return_sequences=1 ,pad_token_id = tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Chatbot Project")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")

