### DONOT DELETE THIS IS FOR LOCAL##################........................

# from langchain.llms import Ollama
# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.prompts import PromptTemplate
# import spacy

# nlp = spacy.load("en_core_web_sm")

# # Initialize the Ollama model
# mixtral_llm = Ollama(model="mistral:7b", base_url="http://localhost:11434")

####### USING API FOR MIXTRAL##########
from langchain_mistralai import ChatMistralAI
from langchain.llms import ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
import spacy

nlp = spacy.load("en_core_web_sm")
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

import os

load_dotenv()

mixtral_llm = ChatMistralAI(
    mistral_api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small", temperature=0.7
)


# prompt for filtering
filter_prompt = PromptTemplate(
    template=(
        """
        You are a legal query classifier. Strictly categorize the following query as either 'Legal' or 'Non-Legal' based on cybercrime relevance:

        **Classify as 'Legal' ONLY if it explicitly involves:**
        1. Unauthorized digital access (hacking, account takeover, system intrusion)
        2. Online financial crimes (scams, phishing, cryptocurrency fraud)
        3. Data compromise (breaches, leaks, doxxing, sensitive info exposure)
        4. Identity-based crimes (theft, impersonation, fake profiles)
        5. Digital harassment (cyberbullying, revenge porn, threats via platforms)
        6. Technology-facilitated crimes (dark web activities, malware distribution)
        7. Specific legal actions/impacts (arrests, lawsuits, regulatory violations)

        **Classify as 'Non-Legal' for:**
        1. General tech issues (password reset, software installs, performance issues)
        2. Basic internet safety questions without specific incidents
        3. Non-cyber legal matters (physical crimes, family law, contracts)
        4. Hypothetical/theoretical scenarios without real-world impact
        5. Platform usage questions (account deletion, feature inquiries)
        6. General tech discussions (AI ethics, cryptocurrency concepts)
        7. Non-actionable complaints (spam emails without financial loss)

        **Decision Guidelines:**
        - Assume malicious intent for account access issues
        - Require explicit mention of harm for financial/identity matters
        - Treat data leaks as legal only if personal/sensitive info involved
        - Ignore jurisdiction considerations - focus on incident nature
        - Default to 'Non-Legal' for ambiguous cases

        **Response Format:**
        Reply ONLY with 'Legal' or 'Non-Legal' - no explanations.

        Query: {input_text}
    """
    ),
    input_variables=["input_text"],
)


# def filter_query(input_text):
#     response = mixtral_llm(filter_prompt.format(input_text=input_text))
#     return response.strip()


def filter_query(input_text):
    prompt_text = filter_prompt.format(input_text=input_text)
    response = mixtral_llm.invoke([HumanMessage(content=prompt_text)])
    return response.content.strip()


# For checking names by regex pattern
def check_for_names(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            return True
    return False


# Function for general conversation
def general_convo(input_text):
    if check_for_names(input_text):
        return "No data available"
    else:
        short_prompt = f"Reply in less than 20 words: {input_text}. Do not translate. No explanations needed."

        response = mixtral_llm(short_prompt)
        return response.strip()


# Instance for general_convo tool
general_convo_instance = Tool(
    name="General_conversation",
    func=general_convo,
    description="Handles non-legal queries with a general conversation model ",
)

# function for legal queries
def legal_convo():
    return "Legal"


# Instance for legal_convo tool
legal_convo_instance = Tool(
    name="Legal_conversation",
    func=legal_convo,
    description="Handles legal queries specifically by returning 'Legal'.",
)

tools = [general_convo_instance, legal_convo_instance]

agent = initialize_agent(
    tools=tools,
    llm=mixtral_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)


def run_agent(user_query):
    query_type = filter_query(user_query)
    if query_type == "Legal":
        response = legal_convo()
        print("Legal Response: ", response)
        return response
    else:
        response = agent.run(user_query)
        print("General Response: ", response)
        return response


def rag_generate(context):
    response = mixtral_llm.invoke(context)

    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)
