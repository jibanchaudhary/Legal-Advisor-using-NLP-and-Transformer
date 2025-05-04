# function.
def get_general_response(query, general_conversations):
    for conversation in general_conversations["general_conversations"]:
        if query.lower() == conversation["query"]:
            return conversation["response"]
    return "I'm not sure how to respond to that. Could you elaborate?"


# yo chai condition hai check garna
if len(query.split()) < 5:
    general_conversations = load_json_file("general_conversation.json")
    response = get_general_response(query, general_conversations)

    request.session["irrelevant_query"] = response
    return redirect("advisory:irrelevant_query")
