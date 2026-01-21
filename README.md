# Legal-Advisor-using-NLP-and-Transformer

Legal Advisor using NLP and Transformer is an AI-powered cyber crime legal advisory system designed to assist users in understanding cyber crime laws, procedures, and legal remedies under Nepal’s Electronic Transactions Act (ETA) 2063.

The system accepts natural language queries, performs legal relevance validation using transformer-based models, retrieves semantically relevant legal documents using Retrieval-Augmented Generation (RAG), and generates concise, structured legal guidance while preserving user privacy.

This project was developed as a Major Project for the Bachelor of Computer Engineering at Advanced College of Engineering and Management (ACEM), Tribhuvan University.

---

## System Pipeline

The system follows a multi-stage NLP pipeline:

- Query ingestion
- Legal relevance classification
- Semantic embedding and document retrieval
- Context-aware response generation

---

## Architecture and Workflow

### Flow Diagram

<img width="846" height="978" alt="flowchart" src="https://github.com/user-attachments/assets/30cb37fc-1a73-44e4-8520-179944db1c8f" />

### Workflow Steps

1. User submits a natural language query
2. Query is validated for legal relevance using a Mistral-based agent
3. Legal queries are embedded using BERT
4. Vector similarity search retrieves top-k relevant documents
5. Retrieved context is passed to the generation module
6. Structured legal response is returned to the user

---

## Models and Methods

### BERT
- Encoder-only transformer architecture
- Fine-tuned on cyber crime legal text
- Used for semantic embedding of queries and documents

### Retrieval-Augmented Generation (RAG)
- Dense vector retrieval over legal documents
- Euclidean distance used for similarity computation

### Mistral 7B
- Large language model used for domain relevance validation
- Prevents non-legal queries from entering the retrieval pipeline

---

## Training and Evaluation

Transformer models were fine-tuned on domain-specific legal data.

### Evaluation Metrics
- Training loss
- Validation loss
- Accuracy

### Training Performance

<img width="1066" height="741" alt="training" src="https://github.com/user-attachments/assets/1f8133b5-ec58-4ced-9a2f-9a713ff0a151" />
<img width="1032" height="727" alt="validation" src="https://github.com/user-attachments/assets/5fcb88c2-3a0d-4199-b348-cffabba072e2" />

## System Output

<img width="4000" height="2548" alt="op-1" src="https://github.com/user-attachments/assets/89ecbd40-d4c1-46aa-813c-1661c4f780a3" />

<img width="2520" height="4000" alt="op-2" src="https://github.com/user-attachments/assets/1367940c-b588-411d-bcbd-55412b15b7e8" />

## Future Work

- Multilingual query support
- Cross-jurisdiction legal datasets
- Approximate nearest neighbor retrieval
- Lightweight model deployment
- Enhanced privacy-preserving inference

---

## Authors

Jiban Chaudhary  
Gauri Shankar Shah  
Nawraj Basnet  

Bachelor of Computer Engineering  
Advanced College of Engineering and Management (ACEM)

## Supervisor
Assoc. Prof. Dr. Nanda Bikram Adhikari  
Associate Professor, Institute of Engineering, Pulchowk Campus, Tribhuvan University
[Google Scholar](https://scholar.google.com/citations?hl=en&user=O8AgVbMAAAAJ)






