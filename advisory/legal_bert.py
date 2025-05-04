from transformers import (BertTokenizer, BertModel, BartTokenizer,
                          BartForConditionalGeneration, AutoModelForMaskedLM,
                          AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer,
                          BertForSequenceClassification)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import json
import re
import faiss
import torch
import pandas as pd
import numpy as np
import os
from django.conf import settings
from .models_loading import *

#sentence embedding
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from .utils import *

# Load Sentence-BERT model for embeddings
semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# Preprocessing legal documents
def preprocess_documents(doc_list):
    return [clean_text(doc.get("title", "") + " " + doc.get("description", "")) + " " + (doc.get["activity",""] if "activity" in doc else "") for doc in doc_list]

#f Filtering out only the English Queries
def validate_query(text):
    # Check if text contains only English letters and spaces
    pattern = r'^[a-zA-Z\s\?\.\,\!\:\;\"\']+$'
    if re.match(pattern, text):
        return text
    return None


# Store embeddings in FAISS Index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieving top_k
def retrieve_top_k(query, index, embeddings, docs, top_k=3):
    query_embedding = semantic_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [(docs[i], distances[0][j]) for j, i in enumerate(indices[0])]



# RAG generation
def generate_incident_summary(incident):
    law_id = incident.get('law_id', 'Law ID not available')
    title = incident.get('title', 'Title not available')
    description = incident.get('description', 'Description not available')
    district = incident.get('location', {}).get('district', 'District not available')
    municipality = incident.get('location', {}).get('municipality', 'Municipality not available')
    ward = incident.get('location', {}).get('ward', 'Ward not available')
    suspect_name = incident.get('suspect', {}).get('name', 'Suspect name not available')
    suspect_age = incident.get('suspect', {}).get('age', 'Suspect age not available')
    suspect_activity = incident.get('suspect', {}).get('activity', 'Suspect activity not available')
    arrest_date = incident.get('arrest_date', 'Arrest date not available')
    legal_action = incident.get('legal_action', 'Legal action not available')
    law_sections = ', '.join(incident.get('law_sections', ['Law sections not available']))

    context = f"""
    Incident Details:
    - Law ID: {law_id}
    - Title: {title}
    - Description: {description}
    - Location: {district}, {municipality}, Ward {ward}
    - Suspect: {suspect_name}, Age: {suspect_age}
    - Suspect Activity: {suspect_activity}
    - Arrest Date: {arrest_date}
    - Legal Action: {legal_action}
    - Law Sections: {law_sections}

    Based on the above legal context, generate a concise yet detailed summary of this case.Donot change the Arrest Date and Ensure the summary integrates the suspect's activity and its implications under the law donot include the word summary at the beginning.
    """

    response = rag_generate(context)
    return response

def generate_policy_summary(policy):
    law_id = policy.get('law_id', 'Law ID not available')  # Safe access
    if 'penalty' in policy:
        penalty_info = policy['penalty']
        penalty_description = f"Fine: {penalty_info['fine']}, Imprisonment: {penalty_info['imprisonment']}, Both: {penalty_info['both']}"
    else:
        penalty_description = "Penalty information not available."

    section_id = policy.get('section_id', 'Section ID not available')
    title = policy.get('title', 'Title not available')
    description = policy.get('description', 'Description not available')

    context = f"""
    Legal Context:
    - Law ID: {law_id}
    - Section: {section_id}
    - Title: {title}
    - Description: {description}
    - Penalty: {penalty_description}

    Based on the above legal context, generate a concise yet detailed summary of this case. Ensure the summary integrates the suspect's activity and its implications under the law donot include the word summary at the beginning.
    """

    response = rag_generate(context)  # Assuming RAG is implemented
    return response



# Label encoding from the dataset
dataset_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/advisory/legal_datasets_expanded_corrected.csv"
df = pd.read_csv(dataset_path)
labels = df.Label.values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# for general conversation
def get_general_response(query, general_conversations):
    for conversation in general_conversations['general_conversations']:
        if query.lower() == conversation['query']:
            return conversation['response']
    return "I'm not sure how to respond to that. Could you elaborate?"


def load_keywords():
    return load_json_file('keywords.json')

# Helper Functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()               
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def generate_embeddings(texts, tokenizer, model):
    combined_texts = [f"{title} {description}" for title, description in texts]
    inputs = tokenizer(combined_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.encoder_last_hidden_state.mean(dim=1).numpy()

####------FOR BM-25 AND TFIDF functions only----####
def find_relevant_data(user_query,label_sentences,incident_corpus, policy_corpus, bm25_incidents, bm25_policies,incident_texts,policies_texts):

    query_text = ' '.join(label_sentences) if isinstance(label_sentences, list) else label_sentences
    query = clean_text(query_text)
    print(query)
    tokenized_query = query.split()  
    print(tokenized_query)

    threshold = 1.2
    combined_query = f"{user_query} {query_text}"
    print(combined_query)
    query_embedding = semantic_model.encode([combined_query], convert_to_numpy=True)[0]
    print(query_embedding)


    incident_embeddings = semantic_model.encode(
    [clean_text(doc) for doc in incident_corpus], convert_to_numpy=True
    )
    print(len(incident_embeddings))

    policy_embeddings = semantic_model.encode(
    [clean_text(doc) for doc in policy_corpus], convert_to_numpy=True
    )
    print(len(policy_embeddings))

    scaler_incident_bm25 = StandardScaler()
    incident_scores = scaler_incident_bm25.fit_transform(
    bm25_incidents.get_scores(tokenized_query).reshape(-1, 1)
    ).flatten()

    # Normalize semantic incident scores
    semantic_incident_scores = np.array([
    1 - cosine(query_embedding, emb) for emb in incident_embeddings
    ])
    scaler_incident_semantic = StandardScaler()
    semantic_incident_scores = scaler_incident_semantic.fit_transform(
    semantic_incident_scores.reshape(-1, 1)
    ).flatten()

# Normalize BM25 policy scores
    scaler_policy_bm25 = StandardScaler()
    policy_scores = scaler_policy_bm25.fit_transform(
    bm25_policies.get_scores(tokenized_query).reshape(-1, 1)
    ).flatten()

# Normalize semantic policy scores
    semantic_policies_scores = np.array([
    1 - cosine(query_embedding, emb) for emb in policy_embeddings
    ])
    scaler_policy_semantic = StandardScaler()
    semantic_policies_scores = scaler_policy_semantic.fit_transform(
    semantic_policies_scores.reshape(-1, 1)
    ).flatten()

    alpha = 0.6  # Give more weight to BM25 for keyword relevance
    final_incident_scores = alpha * incident_scores + (1 - alpha) * semantic_incident_scores
    final_policy_scores = alpha * policy_scores +(1-alpha) * semantic_policies_scores
    print(f"final_incident_scores:{final_incident_scores}")
    print(f"final_policies_scores:{final_policy_scores}")

    best_incident_idx = np.argmax(final_incident_scores)
    print(best_incident_idx)
    best_policy_idx = np.argmax(final_policy_scores)
    print(best_policy_idx)

    best_incident_score = final_incident_scores[best_incident_idx]
    print(best_incident_score)
    best_policy_score = final_policy_scores[best_policy_idx]
    print(best_policy_score)

    if best_incident_score < threshold:
        most_relevant_incident = "No relevant incident found."
    else:
        most_relevant_incident = incident_texts[best_incident_idx]

    # Check the policy score
    if best_policy_score < threshold:
        most_relevant_policy = "No relevant policy found."
    else:
        most_relevant_policy = policies_texts[best_policy_idx]


    return most_relevant_incident, most_relevant_policy


def match_crime_type(input_text):
    input_text = input_text.lower()
    keywords = load_keywords()
    for label in keywords["labels"]:
        label = label.lower()
        if input_text == label:
            print("Match found")
            return label
    print("Match not found")
    return None

def generate_labels(query,tokenizer,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded_query = tokenizer.encode_plus(
        query,
        add_special_tokens= True,
        max_length = 128,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    input_ids = encoded_query['input_ids'].to(device)
    attention_mask = encoded_query['attention_mask'].to(device)

    model.eval()
    # get_predictions
    with torch.no_grad():
        outputs = model(input_ids,attention_mask=attention_mask)
        logits = outputs.logits
    #Converting logits to predicted label
    predicted_label_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    # Print the result
    print(f"Query: {query}")
    print(f"Predicted Label: {predicted_label}")

    return predicted_label

# Extracting the sentences associated with the labels from the label_sentence dictionary
def extract_sentences(label, keywords):
    return keywords["label_sentences"].get(label,[])

# Getting bert embedding
# def get_bert_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt")
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# # using bertembedding to get the similarity value
# def generate_recommendations_bert(crime_type, tokenizer, model, crime_recommendations):
#     if crime_type:
#         input_embedding = get_bert_embedding(crime_type, tokenizer, model)
#         crime_embeddings = {k: get_bert_embedding(k, tokenizer, model) for k in crime_recommendations.keys()}
#         similarities = {k: cosine_similarity(input_embedding, v)[0][0] for k, v in crime_embeddings.items()}
#         best_match = max(similarities, key=similarities.get)
#         return crime_recommendations[best_match]
#     else:
#         return '''No specific recommendations are available.
#             Contact Bureau office: +977 9851286770, +977-01-5319044
#             Facebook: https://facebook.com/cyberbureaunepal
#             Email: cyberbureau@nepalpolice.gov.np'''

#Find crimetype
def find_category(crime_type, dataset_keywords):
    # Loop through the dataset to find a matching category based on the crime type keyword
    for category, keywords in dataset_keywords.items():
        if crime_type.lower() in [keyword.lower() for keyword in keywords]:
            return category
    return None

def generate_recommendations_bert(category, procedure_data):
    # Search the procedure.json for the title matching the category
    if category in procedure_data:
        return procedure_data[category]
    return '''No specific recommendations are available.
              Contact Bureau office: +977 9851286770, +977-01-5319044
              Facebook: https://facebook.com/cyberbureaunepal
              Email: cyberbureau@nepalpolice.gov.np'''


#formatting the recommendation part
def format_recommendations(recommendations):
    if isinstance(recommendations, list):
        lines = recommendations
    elif isinstance(recommendations, str):
        # Split based on numbered sections and maintain their order
        lines = re.split(r'(?<=\.)\s*(?=\d+\s)', recommendations.strip())
    else:
        return recommendations

    formatted = ''
    for line in lines:
        line = line.replace('..', '.').strip()
        if line.startswith(('1', '2', '3', '4', '5')):  # Adjust to handle more numbers if needed
            formatted += f'\n{line.strip()}\n'
        else:
            formatted += f'  {line.strip()}\n'

    print(str(formatted))

    return formatted

# def load_fine_tuned_model(output_dir='/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/advisory/combined_modelGPT2-16500'):
#     if output_dir:
#         print(f"Loading fine-tuned GPT-2 model from {output_dir}...")
#         model = GPT2LMHeadModel.from_pretrained(output_dir)
#         tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
#     else:
#         print("Loading the original GPT-2 model...")
#         model = GPT2LMHeadModel.from_pretrained("gpt2")  
#         tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
    
#     return model, tokenizer

# conversation_history = ""

# def generate_response(user_input ,max_length=100):
#     global conversation_history

#     gpt_model, gpt_tokenizer = get_gpt_2_models_and_tokenizer()

#     conversation_history += f"You: {user_input}\n Chatbot: "

#     inputs = gpt_tokenizer.encode(conversation_history, return_tensors="pt")
#     outputs = gpt_model.generate(
#         inputs,
#         max_length=max_length + len(inputs[0]),
#         num_return_sequences=1,
#         no_repeat_ngram_size=2,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7,
#     )

#     response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     chatbot_reply = response[len(conversation_history):].strip()

#     chatbot_reply = chatbot_reply.split(".")[0].strip()
#     conversation_history += f"{chatbot_reply}\n"
#     return chatbot_reply



# Load datasets
def load_json_file(file_name):
    file_path = os.path.join(settings.BASE_DIR, 'documents', 'data', file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return []
    
# # Loading the fine-tuned GPT-2 model
# gpt_model = None
# gpt_tokenizer = None

# def get_gpt_2_models_and_tokenizer():
#     global gpt_model, gpt_tokenizer
#     if gpt_model is None or gpt_tokenizer is None:
#         raise RuntimeError("GPT-2 model and tokenizer are not loaded. Check the `apps.py` configuration.")
#     return gpt_model, gpt_tokenizer

# Initialize models and tokenizers
# bert_model = BertForSequenceClassification.from_pretrained("/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/advisory/fine_tuned_legal_bert")
# bert_tokenizer = BertTokenizer.from_pretrained("/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/advisory/fine_tuned_legal_bert")

# bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")



