import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Specify the directory path containing the PDF files
sample_policies_folder = './sample_policies'

# Get a list of all PDF files in the directory
pdf_files = [file for file in os.listdir(sample_policies_folder) if file.endswith('.pdf')]

documents = []  # Initialize an empty list for storing the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Loop through each PDF file
for pdf_file in pdf_files:
    pdf_file_path = os.path.join(sample_policies_folder, pdf_file)
    print(f"Processing PDF file: {pdf_file_path}")

    # Open the PDF file
    with open(pdf_file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Extract the text from each page
        document_text = ''
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text = page.extract_text()
            document_text += text

        # Create a file path for saving the extracted text
        text_file_name = os.path.splitext(pdf_file)[0] + '.txt'
        text_file_path = os.path.join(sample_policies_folder, text_file_name)

        # Create a document with the appropriate metadata
        document = Document(page_content=document_text, metadata={'source': pdf_file})
        documents.append(document)  # Append the document to the list of documents

        # Split the document into smaller texts
        texts = text_splitter.split_documents([document])

        # Access the individual text chunks
        for text in texts[0]:
            print(text)
            print(document.metadata)  # Print the metadata for the current document
            print('\n')

        print(f"Extracted text saved to: {text_file_path}\n")

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = './db'
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

# Persist the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk and use it as normal.
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Set up the turbo LLM
turbo_llm = OpenAI(model_name='gpt-3.5-turbo')

# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff", retriever=retriever,
                                       return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        print('\n')
