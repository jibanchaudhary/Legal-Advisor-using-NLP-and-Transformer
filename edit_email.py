# Libraries
import spacy
import re
from transformers import BartTokenizer,BartForConditionalGeneration


# Read content
with open("output.txt",'r') as file:
    raw_content = file.read()

#Load the model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# BART model for text editing
def bart_edit(content):
    inputs = tokenizer.encode(content, return_tensors = "pt", max_length = 1024, truncation = True)

    # generate text
    outputs = model.generate(inputs,max_length = 1000, num_beams = 4, early_stopping = True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return generated_text

# mail template
def report_template(subject, reporter_name, edited_text):
    template = f"""
    To: Cyber Bureau
    
    Subject: {subject}

    Dear Sir/Madam,

    I am writing to formally report a cybercrime incident that I have encountered. Below are the details of the incident for your kind attention and necessary action:

    Reported By:
    {reporter_name}

    Incident Details:
    {edited_text}

    I kindly request your assistance in investigating and resolving this matter. Please let me know if further details or documents are required to support the investigation. I am willing to cooperate fully to bring the matter to justice.

    Thank you for your time and attention.

    Yours sincerely,
    {reporter_name}
    """
    return template

#Loading spacy nlp model
nlp = spacy.load("en_core_web_sm")

# Extracting name
def extract_name(raw_content):
    match = re.search(r"My name is ([A-Z][a-z]*)[\s-]([A-Z][a-z]*)", raw_content)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

subject_Keywords=set([
    # Financial Fraud
    "scam", "fraud", "phishing", "unauthorized transactions", "money laundering","Financial scam",
    "credit card fraud", "fake investments", "Ponzi scheme", "pyramid scheme",
    "identity theft", "account hacking", "fake lottery", "loan fraud",
    
    # Online Shopping Scams
    "fake website", "counterfeit goods", "non-delivery", "refund scam", "payment fraud",
    "online marketplace scam",
    
    # Social Engineering and Impersonation
    "impersonation", "fake profiles", "social media scam", "WhatsApp scam",
    "email fraud", "vishing", "smishing",
    
    # Tech Support Scams
    "fake tech support", "remote access fraud", "computer virus scam", "pop-up scam",
    
    # Cryptocurrency Scams
    "crypto scam", "fake ICO", "wallet fraud", "Bitcoin fraud", "cryptocurrency hacking",
    
    # Employment and Job Scams
    "fake job offer", "recruitment fraud", "work-from-home scam", "freelance fraud",
    
    # Investment Scams
    "stock fraud", "forex scam", "investment scheme", "fake broker",
    
    # Romance Scams
    "online dating scam", "relationship fraud", "catfishing", "love scam",
    
    # Data and Privacy Violations
    "personal data theft", "unauthorized access", "credential theft", "data breach",
    
    # Miscellaneous Scams
    "insurance fraud", "fake charity", "ransom demand", "extortion", "fake loan",
    "contest fraud"
])
def extracts_template_field(raw_content):
    doc = nlp(raw_content)
    subject, reporter_name = None,None

    #Extraction for subject
    for sent in doc.sents:
        for keyword in subject_Keywords:
            if keyword in sent.text.lower():
                subject = keyword
                break
        if subject:
            break

    #Extraction for reporter name
    reporter_name = extract_name(raw_content)
    if reporter_name is None:
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not reporter_name:
                reporter_name = ent.text


    return{
        "subject":subject or "General Cybercrime Report",
        "reporter_name" : reporter_name
    }




# Output
fields = extracts_template_field(raw_content)
edited_text = bart_edit(raw_content)
output = report_template(fields["subject"],fields["reporter_name"],edited_text)


# storing the text
with open(f"email_{fields['reporter_name']}.txt","a") as f:
    f.write(output)
