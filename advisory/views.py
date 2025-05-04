from django.shortcuts import render, redirect
from .legal_bert import *
from .models import *
from .models_loading import *
from django.http import JsonResponse
import json
import faiss
from rank_bm25 import BM25Okapi
from .utils import *

# render index.html
from django.shortcuts import render


def index(request):
    return render(request, "index.html")


# labels for company keywords in match_crime_type function
dataset_path = "/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/advisory/legal_datasets.csv"
df = pd.read_csv(dataset_path)
labels = df.Label.values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


def advisory_query(request):
    if request.method == "POST":
        query = request.POST.get("query", "")
        query = query.lower()
        query = validate_query(query)

        if not query:
            return JsonResponse(
                {
                    "error": "Oops! It looks like you forgot to enter a query. Please provide one and try again."
                }
            )

        # if len(query.split()) < 5:
        #     general_conversations = load_json_file('general_conversation.json')
        #     response = get_general_response(query, general_conversations)

        #     request.session['irrelevant_query'] = response
        #     return redirect('advisory:irrelevant_query')
        response = run_agent(query)
        print(f"run_agent output: {response}")  # Debugging print

        if response.lower() != "legal":
            # Handle non-legal response
            request.session["irrelevant_query"] = response
            return redirect("advisory:irrelevant_query")
        else:
            # labeling the query from generate_label funciton from legal_bert
            label = generate_labels(query, legal_bert_tokenizer, legal_bert_model)
            label = str(label)
            keywords = load_keywords()
            label_sentences = extract_sentences(label, keywords)

            print(label_sentences)

            # load datafiles
            if match_crime_type(label):

                incidents = load_json_file("incidents.json")
                policies = load_json_file("policies.json")
                crime_recommendations = load_json_file("procedures.json")
                procedure_keywords = load_json_file("procedure_keywords.json")

                # generating incident and policy corpus
                incident_corpus = preprocess_documents(incidents)
                print("Preprocessing incident corpus success")
                policy_corpus = preprocess_documents(policies)
                print("Preprocessing policy corpus success")

                # Semantic model to generate embeddings
                incident_embeddings = semantic_model.encode(
                    incident_corpus, convert_to_numpy=True
                )
                policy_embeddings = semantic_model.encode(
                    policy_corpus, convert_to_numpy=True
                )

                # Extracting the incident and policy index
                incident_index = create_faiss_index(incident_embeddings)
                policy_index = create_faiss_index(policy_embeddings)

                # Retrieving top_k results
                incident_results = retrieve_top_k(
                    query, incident_index, incident_embeddings, incident_corpus
                )
                policy_results = retrieve_top_k(
                    query, policy_index, policy_embeddings, policy_corpus
                )

                # first index as title and second as description
                thresold = 2.0
                best_incident = None
                for doc, score in incident_results:
                    if score < thresold:
                        for incident in incidents:
                            if (
                                incident["title"].lower() in doc
                                or incident["description"].lower() in doc
                            ):
                                best_incident = incident
                                break  # Exit the inner loop immediately when a match is found
                    if best_incident:
                        break  # Exit the outer loop immediately when a match is found
                print(best_incident)
                best_policy = None
                for doc, score in policy_results:
                    if score < thresold:
                        for policy in policies:
                            if (
                                policy["title"].lower() in doc
                                or policy["description"].lower() in doc
                            ):
                                best_policy = policy
                                break  # Exit the inner loop immediately when a match is found
                    if best_policy:
                        break  # Exit the outer loop immediately when a match is found
                print(best_policy)

                # calline generate summary in legal_bert.py
                summary_incident = generate_incident_summary(best_incident)
                summary_policy = generate_policy_summary(best_policy)

                # Recommendation using BERT
                category = find_category(label, procedure_keywords)
                bert_recommendations = generate_recommendations_bert(
                    category, crime_recommendations
                )
                bert_recommendations_formatted = format_recommendations(
                    bert_recommendations
                )
                context = {
                    "incident": {
                        "title": best_incident["title"]
                        if best_incident
                        else "No incident found",
                        "description": summary_incident
                        if best_incident
                        else "No description available",
                    },
                    "policy": {
                        "title": best_policy["title"]
                        if best_policy
                        else "No policy found",
                        "description": summary_policy
                        if best_policy
                        else "No description available",
                    },
                    "recommendations": bert_recommendations_formatted,
                }

                request.session["query_result"] = context

            else:
                # for external chatbot model reply
                request.session["query_result"] = {"response": ""}
            return redirect("advisory:query_result")
    return render(request, "query_form.html")


def query_result(request):
    context = request.session.get("query_result", {})
    return render(request, "query_result.html", context)


def irrelevant_query(request):
    response = request.session.get("irrelevant_query", "No response available.")
    return render(request, "irrelevant_query.html", {"response": response})
