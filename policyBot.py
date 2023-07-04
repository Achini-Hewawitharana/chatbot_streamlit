import os
import subprocess
import sys

# Define the required packages
required_packages = [
    "streamlit",
    "PyPDF2",
    "langchain",
    "openai",
    # "pdfplumber",
    # "panel"
]

# Check if packages are already installed
installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
packages_to_install = [package for package in required_packages if package not in installed_packages]

# Install required packages if they are not already installed
if packages_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit as st

import openai
# import pdfplumber
# import panel as pn


# Specify the directory path containing the PDF files
sample_policies_folder = './sample_policies'

# Set OpenAI API key
# openai.api_key = ''

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Helper function to get chatbot response
def get_chatbot_response(user_input, context):
    messages = [{'role': 'system', 'content': context}, {'role': 'system', 'content': user_input}]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message['content']

# Streamlit app code
def main():
    st.title('Public Policy Chatbot')

    # User input
    user_input = st.text_input('User Input')

    # Send button
    if st.button('Send'):
        if user_input:
            # Get chatbot response with context
            context = "You are a friendly and helpful Policy Bot"
            chatbot_response = get_chatbot_response(user_input, context)

            # Display chatbot response
            st.text_area('Chatbot Response', value=chatbot_response, height=200)

    # File upload
    uploaded_files = st.file_uploader('Upload PDF(s)', type='pdf', accept_multiple_files=True)

    # Process uploaded files and display extracted text
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)

            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                extracted_text = page.extract_text()
                st.text_area(f'Extracted Text (Page {page_number+1})', value=extracted_text, height=200)

if __name__ == '__main__':
    main()
