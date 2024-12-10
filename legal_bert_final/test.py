from transformers import (BertTokenizer, BertModel, BartTokenizer
                          , BartForConditionalGeneration,AutoModelForMaskedLM,AutoTokenizer
                          ,GPT2LMHeadModel, GPT2Tokenizer)
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import torch


# Helper Functions
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def generate_embeddings(texts, tokenizer, model):
    combined_texts = [f"{title} {description}" for title, description in texts]
    inputs = tokenizer(combined_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.encoder_last_hidden_state.mean(dim=1).numpy()

def find_relevant_data(query, incident_texts, policy_texts, incident_embeddings, policy_embeddings, tokenizer, model):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        query_embedding = model(**inputs).encoder_last_hidden_state.mean(dim=1).numpy()

    incident_similarities = cosine_similarity(query_embedding, incident_embeddings)
    policy_similarities = cosine_similarity(query_embedding, policy_embeddings)

    best_incident_idx = incident_similarities.argmax()
    best_policy_idx = policy_similarities.argmax()

    most_relevant_incident = incident_texts[best_incident_idx]
    most_relevant_policy = policy_texts[best_policy_idx]

    return most_relevant_incident, most_relevant_policy


def match_crime_type(input_text):
    for crime_type, words in keywords.items():
        if any(word in input_text.lower() for word in words):
            return crime_type
    return None

# Generates procedures
def generate_recommendations_bart(crime_type, tokenizer, model, crime_recommendations):
    recommendations = crime_recommendations.get(crime_type)
    if not recommendations:
        return '''No specific recommendations are available.
            Contact Bureau office: +977 9851286770, +977-01-5319044
            Facebook: https://facebook.com/cyberbureaunepal
            Email: cyberbureau@nepalpolice.gov.np'''
    
    input_text = " ".join(recommendations)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Loading pretrained chatbot model
def load_fine_tuned_model(output_dir = "/Users/jibanchaudhary/Documents/Projects/Gmail integration/fine_tuned_gpt"):
    print("Loading pretrained gpt2 model....")
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    return model, tokenizer

conversation_history = ""

# Chatbot response 
def generate_response(user_input, model, tokenizer, max_length =100):
    global conversation_history

    conversation_history += f"You: {user_input}\n Chatbot: "

    #Encoding
    inputs = tokenizer.encode(conversation_history, return_tensors = "pt")
    outputs = model.generate(
        inputs,
        max_length = max_length + len(inputs[0]),
        num_return_sequences = 1,
        no_repeat_ngram_size = 2,
        top_k = 50,
        top_p = 0.95,
        temperature = 0.7,
    )

    # Decoding
    response = tokenizer.decode(outputs[0],skip_special_tokens= True)
    chatbot_reply = response[len(conversation_history):].strip()

    chatbot_reply = chatbot_reply.split(".")[0].strip()
    conversation_history+=f"{chatbot_reply}\n"
    return chatbot_reply


# Main Logic
if __name__ == "__main__":
    # loading the gpt model and tokenizer
    gpt_model, gpt_tokenizer = load_fine_tuned_model()

    # Load datasets
    def load_json_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return []

    incidents = load_json_file('incidents.json')
    policies = load_json_file('policies.json')
    crime_recommendations = load_json_file('procedures.json')
    keywords = load_json_file('keywords.json')

    # Initialize models and tokenizers
    bert_model = BertModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    bert_tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Preprocess incidents and policies
    incident_texts = [
        (clean_text(item.get('title', '')), clean_text(item.get('description', '')))
        for item in incidents
    ]
    policy_texts = [
        (clean_text(item.get('title', '')), clean_text(item.get('description', '')))
        for item in policies
    ]

    # Generate embeddings
    incident_embeddings = generate_embeddings(incident_texts, bart_tokenizer, bart_model)
    policy_embeddings = generate_embeddings(policy_texts, bart_tokenizer, bart_model)

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "exit":
            print("Good bye")
            break

        # In case of Legal conversation
        if match_crime_type(user_query):
            relevant_incident, relevant_policy = find_relevant_data(
        user_query, incident_texts, policy_texts, incident_embeddings, policy_embeddings, bart_tokenizer, bart_model)

            print("\nMost Relevant Incident:")
            print(f"Title: {relevant_incident[0]}\nDescription: {relevant_incident[1]}")

            print("\nMost Relevant Policy:")
            print(f"Title: {relevant_policy[0]}\nDescription: {relevant_policy[1]}")

            # Recommendations
            crime_type_input = input("\nEnter a crime type: ")
            matched_crime_type = match_crime_type(crime_type_input)
            if matched_crime_type:
                recommendations = generate_recommendations_bart(
                    matched_crime_type, bart_tokenizer, bart_model, crime_recommendations
                )
                print("\nRecommendations:")
                print(recommendations)
            else:
                print("\nNo matching crime type found.")
        else:
            # In case of General conversation
            response = generate_response(user_query,gpt_model,gpt_tokenizer)
            print(f"Chatbot: {response}")
            

