import json,os
from django.conf import settings

def get_general_response(query,general_conversations):
    for conversation in general_conversations["general_conversations"]:
        if query == conversation["query"]:
            return conversation["response"]


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Parse JSON content into a Python dictionary
            return data
    except Exception:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    
query=input("Enter the query")
general_conversations = load_json("/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/documents/data/general_conversation.json")
output=get_general_response(query,general_conversations)
print(output)
