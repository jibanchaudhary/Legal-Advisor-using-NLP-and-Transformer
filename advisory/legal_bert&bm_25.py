from transformers import (
    BertTokenizer,
    BertModel,
    BartTokenizer,
    BartForConditionalGeneration,
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BertForSequenceClassification,
)
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import LabelEncoder
import json
import re
import torch
import pandas as pd
import numpy as np

# Helper Functions
def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


def generate_embeddings(texts, tokenizer, model):
    combined_texts = [f"{title} {description}" for title, description in texts]
    inputs = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.encoder_last_hidden_state.mean(dim=1).numpy()


def find_relevant_data(
    label_sentences, incident_texts, policy_texts, bm25_incidents, bm25_policies
):
    query = clean_text(label_sentences)
    tokenized_query = query.split()

    incident_scores = bm25_incidents.get_scores(tokenized_query)
    policy_scores = bm25_policies.get_scores(tokenized_query)

    best_incident_idx = np.argmax(incident_scores)
    best_policy_idx = np.argmax(policy_scores)

    most_relevant_incident = incident_texts[best_incident_idx]
    most_relevant_policy = policy_texts[best_policy_idx]

    return most_relevant_incident, most_relevant_policy


# Matches the keywords
def match_crime_type(input_text):
    input_text = input_text.lower()
    for label in keywords["labels"]:
        label = label.lower()
        if input_text == label:
            print("Match found")
            return label
    print("Match not found")
    return None


# Generate labels
def generate_labels(query, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded_query = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded_query["input_ids"].to(device)
    attention_mask = encoded_query["attention_mask"].to(device)

    model.eval()
    # get_predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    # Converting logits to predicted label
    predicted_label_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    # Print the result
    print(f"Query: {query}")
    print(f"Predicted Label: {predicted_label}")

    return predicted_label


# Extracting the sentences associated with the labels from the label_sentence dictionary
def extract_sentences(label, keywords):
    return keywords["label_sentences"].get(label, [])


# Generates procedures
def generate_recommendations_bart(crime_type, tokenizer, model, crime_recommendations):
    recommendations = crime_recommendations.get(crime_type)
    if not recommendations:
        return """No specific recommendations are available.
            Contact Bureau office: +977 9851286770, +977-01-5319044
            Facebook: https://facebook.com/cyberbureaunepal
            Email: cyberbureau@nepalpolice.gov.np"""

    input_text = " ".join(recommendations)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Loading pretrained gpt2 chatbot model
def load_fine_tuned_model(output_dir="./combined_modelGPT2-16500"):
    if output_dir:
        print(f"Loading fine-tuned GPT-2 model from {output_dir}...")
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    else:
        print("Loading the original GPT-2 model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


conversation_history = ""

# Chatbot response
def generate_response(user_input, model, tokenizer, max_length=100):
    global conversation_history

    conversation_history += f"You: {user_input}\n Chatbot: "

    # Encoding
    inputs = tokenizer.encode(conversation_history, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length + len(inputs[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    # Decoding
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chatbot_reply = response[len(conversation_history) :].strip()

    chatbot_reply = chatbot_reply.split(".")[0].strip()
    conversation_history += f"{chatbot_reply}\n"
    return chatbot_reply


# Main Logic
if __name__ == "__main__":

    # label_Encoding from the dataset
    dataset_path = (
        "/Users/jibanchaudhary/Documents/Projects/legal_bert/legal_datasets.csv"
    )
    df = pd.read_csv(dataset_path)
    labels = df.Label.values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # loading the gpt model and tokenizer
    gpt_model, gpt_tokenizer = load_fine_tuned_model()

    # Load datasets
    def load_json_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return []

    incidents = load_json_file("incidents.json")
    policies = load_json_file("policies.json")
    crime_recommendations = load_json_file("procedures.json")
    keywords = load_json_file("keywords.json")

    # Initialize models and tokenizers
    bert_model = BertForSequenceClassification.from_pretrained(
        "./fine_tuned_legal_bert"
    )
    bert_tokenizer = BertTokenizer.from_pretrained("./fine_tuned_legal_bert")

    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Preprocess incidents and policies
    incident_texts = [
        (clean_text(item.get("title", "")), clean_text(item.get("description", "")))
        for item in incidents
    ]
    policy_texts = [
        (clean_text(item.get("title", "")), clean_text(item.get("description", "")))
        for item in policies
    ]

    # Tokenize for BM25
    incident_corpus = [" ".join(item) for item in incident_texts]
    policy_corpus = [" ".join(item) for item in policy_texts]

    bm25_incidents = BM25Okapi([doc.split() for doc in incident_corpus])
    bm25_policies = BM25Okapi([doc.split() for doc in policy_corpus])

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "exit":
            print("Good bye")
            break
        else:
            # Calling the label from the generating_label function
            label = generate_labels(user_query, bert_tokenizer, bert_model)

        label = str(label)
        label_sentences = extract_sentences(label, keywords)
        label_sentences = str(label_sentences)

        # In case of Legal conversation find_relevant_data

        if match_crime_type(label):
            print(label_sentences)
            relevant_incident, relevant_policy = find_relevant_data(
                label_sentences,
                incident_corpus,
                policy_corpus,
                bm25_incidents,
                bm25_policies,
            )

            print("\nMost Relevant Incident:")
            print(
                f"Title: {relevant_incident.split(' ')[0]}\nDescription: {' '.join(relevant_incident.split(' ')[1:])}"
            )

            print("\nMost Relevant Policy:")
            print(
                f"Title: {relevant_policy.split(' ')[0]}\nDescription: {' '.join(relevant_policy.split(' ')[1:])}"
            )

            # Recommendations
            crime_type_input = label
            matched_crime_type = match_crime_type(crime_type_input)
            if matched_crime_type:
                recommendations = generate_recommendations_bart(
                    matched_crime_type,
                    bart_tokenizer,
                    bart_model,
                    crime_recommendations,
                )
                print("\nRecommendations:")
                print(recommendations)
            else:
                print("\nNo matching crime type found.")
        else:
            # In case of General conversation
            response = generate_response(user_query, gpt_model, gpt_tokenizer)
            print(f"Chatbot: {response}")
