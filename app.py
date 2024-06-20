import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #Supresses info msges
import numpy as np
import PyPDF2 
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import streamlit as st

model_name= 'bert-large-uncased-whole-word-masking-finetuned-squad'

tokenizer= BertTokenizer.from_pretrained(model_name)
model= BertForQuestionAnswering.from_pretrained(model_name)

# Reading the text from doc
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

def chunk_text(text, tokenizer, max_length=400):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
    return chunks

def bert_find_answer(query, passage):
    inputs= tokenizer(query, passage, return_tensors='pt')
    outputs= model(**inputs)
    sep_index = torch.where(inputs['input_ids'] == tokenizer.sep_token_id)[1].tolist()[0]
    start_scores= outputs.start_logits
    end_scores= outputs.end_logits
    
    start_scores= start_scores.detach().numpy().flatten()
    end_scores = end_scores.detach().numpy().flatten()
    
    # most likely start and end positions
    answer_start= np.argmax(start_scores)
    answer_end= np.argmax(end_scores)+60
    
    best_start_score= np.round(start_scores[answer_start],2) 
    best_end_score= np.round(end_scores[np.argmax(end_scores)],2)
    
    answer= tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))    
        
    return best_start_score, best_end_score, answer

def get_best_answer(query, text):
    sections= chunk_text(text, tokenizer)
    best_answer= ""
    best_score= -float('inf')
    
    for section in sections:
        start_score, end_score, answer= bert_find_answer(query, section)
        if start_score+end_score > best_score:
            best_score= start_score+end_score
            print(best_score)
            best_answer= answer
    
    return best_answer


st.title("PDF Question Answering System")
st.write("Upload a PDF document and ask a question.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your question:")

if uploaded_file is not None and query:
    doc_text = preprocess_text(extract_text_from_pdf(uploaded_file))
    answer = get_best_answer(query, doc_text)
    st.write("Answer:", answer)



