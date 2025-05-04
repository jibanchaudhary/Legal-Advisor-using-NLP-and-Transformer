import speech_recognition as sr
from deep_translator import GoogleTranslator
from transformers import BartTokenizer, BartForConditionalGeneration
import spacy
import re
from .Google import Create_Service
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from advisory.utils import *

# Speech-to-Text
def record_text():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening in Nepali...")
            audio = r.listen(source)
            mytext = r.recognize_google(audio, language="ne-NP")
            print(f"Recognized: {mytext}")
            return mytext
    except sr.RequestError as e:
        print(f"Error: {e}")
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    return None

# Nepali-to-English Translation
def nepali_trans(nepali_text):
    if nepali_text:
        translated_text = GoogleTranslator(source='ne', target='en').translate(nepali_text)
        print(f"Translated text: {translated_text}")
        return translated_text
    return None

# Save Text to File
def output_text(text, filename="output.txt"):
    if text:
        with open(filename, "a") as f:
            f.write(text + "\n")
        print(f"Text saved to {filename}")

# Text Summarization Using BART
def bart_edit(content):
    return(rag_generate(content))

# Generate Email Report Template
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

# Extract Fields from Text Using SpaCy
def extracts_template_field(raw_content):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_content)
    subject_keywords = {
        "scam", "fraud", "phishing", "unauthorized transactions", "identity theft"
    }
    subject, reporter_name = None, None

    # Extraction for subject
    for sent in doc.sents:
        for keyword in subject_keywords:
            if keyword in sent.text.lower():
                subject = keyword
                break
        if subject:
            break

    # Extraction for reporter name
    match = re.search(r"My name is ([A-Z][a-z]*)[\s-]([A-Z][a-z]*)", raw_content)
    if match:
        reporter_name = f"{match.group(1)} {match.group(2)}"
    else:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                reporter_name = ent.text

    return {
        "subject": subject or "General Cybercrime Report",
        "reporter_name": reporter_name,
    }

# Send Email via Gmail API
def send_email(content, subject, recipient_email, reporter_name):
    CLIENT_SECRET_FILE = '/Users/jibanchaudhary/Documents/Projects/legal_assistance/legal_advisory_system/email_integration/client_secret.json'
    API_NAME = 'gmail'
    API_VERSION = 'v1'
    SCOPES = ['https://mail.google.com/']

    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, *SCOPES)
    mime_message = MIMEMultipart()
    mime_message['to'] = 'jiban.077bct042@acem.edu.np'
    mime_message['subject'] = subject
    mime_message.attach(MIMEText(content, 'plain'))
    raw_string = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
    message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
    return message


def generate_email_summary(policy):
    context = f"""
    {policy}
    
    Based on the above context, generate a concise yet detailed legal summary of this case no unnecessary words.
    """

    response = rag_generate(context)
    return response