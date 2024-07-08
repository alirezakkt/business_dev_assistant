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


    outputs = model.generate(input_ids,
     attention_mask = attention_mask, 
     max_length=50,
     min_length=10,
      num_return_sequences=1 ,
      pad_token_id = tokenizer.eos_token_id,
      #temperature = 0.7,# handel creativity of chatbot 
      #top_p = 0.9 ,#nucleus sampling(help to repat answers)
      top_k=50, # decreas token selection
      do_sample=False, #for active sampeling
      num_beams=5, #use beam search
      repetition_penalty=3.0, # stop bot for repeting answers
      no_repeat_ngram_size=2 , # dont generate duplicate tokens
      length_penalty=2.0 , # force the model to generate shorter Resp

      )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    stop_sequence='\n'
    if stop_sequence in response:
        response = response.split(stop_sequence) [0]

    return response

if __name__ == "__main__":
    print("Chatbot Project")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")

