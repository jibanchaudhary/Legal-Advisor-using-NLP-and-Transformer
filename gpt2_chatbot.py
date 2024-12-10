# Libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned GPT-2 model and tokenizer
def load_fine_tuned_model(output_dir="./fine_tuned_gpt"):
    print("Loading fine-tuned GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    return model, tokenizer

# Generate chatbot responses
conversation_history = ""  

def generate_response(user_input, model, tokenizer, max_length=100):
    global conversation_history
    
    conversation_history += f"You: {user_input}\nChatbot: "
    
    # Encode the conversation history
    inputs = tokenizer.encode(conversation_history, return_tensors="pt")
    
    # Generate the response
    outputs = model.generate(
        inputs,
        max_length=max_length + len(inputs[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    
    # Decode and extract the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chatbot_reply = response[len(conversation_history):].strip()

    # Post-process to limit verbosity
    chatbot_reply = chatbot_reply.split(".")[0].strip()  
    conversation_history += f" {chatbot_reply}\n" 
    return chatbot_reply

# Chat loop
def chat():
    print("Chatbot: Hello! I'm your GPT-2 chatbot. Type 'exit' to end the conversation.")
    model, tokenizer = load_fine_tuned_model()
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        try:
            # Generate and print the chatbot's response
            response = generate_response(user_input, model, tokenizer)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat()
