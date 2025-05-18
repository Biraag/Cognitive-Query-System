import streamlit as st
import os
import textwrap
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import io
from docx import Document
import pandas as pd
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API Key not found. Please ensure you have a .env file with GOOGLE_API_KEY.")
    st.stop()

# Configure Google Generative AI with the API key
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {str(e)}")
    st.stop()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to get Gemini response for text input with chat history
def get_gemini_response(question):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # If there's chat history, use it to create a more contextualized response
        if st.session_state.chat_history:
            # Build conversation history in the format Gemini expects
            history = []
            for message in st.session_state.chat_history[:-1]:  # Exclude the latest user message
                if message["role"] == "user":
                    history.append({"role": "user", "parts": [message["content"]]})
                else:
                    history.append({"role": "model", "parts": [message["content"]]})
            
            # Create chat with history
            chat = model.start_chat(history=history)
            
            # Send the current question
            response = chat.send_message(question)
        else:
            # No history, just get a direct response
            response = model.generate_content(question)
            
        return response.text
        
    except Exception as e:
        logger.error(f"Error in get_gemini_response: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Function to get Gemini response for image input
def get_gemini_response_image(input, image):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        if input != "":
            response = model.generate_content([input, image])
        else:
            response = model.generate_content(image)
        return response.text
    except Exception as e:
        logger.error(f"Error in get_gemini_response_image: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Function to get text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error in get_pdf_text: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

# Function to get chunks of text
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error in get_text_chunks: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Function to get a conversational chain
def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the provided context, say, "answer is not available in the context".
        
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logger.error(f"Error in get_conversational_chain: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Function to handle user input for PDF questions
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        if not chain:
            return "Failed to initialize the conversational chain."
            
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        logger.error(f"Error in user_input: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Function to extract text from a Word document
def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)
    except Exception as e:
        logger.error(f"Error in extract_text_from_docx: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

# Function to get response from the Gemini model for Word documents
def get_gemini_response_word(input_text, document_text):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        full_input = f"{input_text}\n\n{document_text}" if input_text else document_text
        response = model.generate_content([full_input])
        return response.text
    except Exception as e:
        logger.error(f"Error in get_gemini_response_word: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Function to read spreadsheet and return its content
def read_spreadsheet(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return None
        return df
    except Exception as e:
        logger.error(f"Error in read_spreadsheet: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Function to get response from the Gemini model for spreadsheets
def get_gemini_response_spreadsheet(input_text, data_frame):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        if input_text:
            # Convert the DataFrame to a string to include in the query
            data_summary = data_frame.to_string(index=False)
            combined_input = f"{input_text}\n\nData:\n{data_summary}"
            response = model.generate_content([combined_input])
            return response.text
        return "No input provided."
    except Exception as e:
        logger.error(f"Error in get_gemini_response_spreadsheet: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Initialize the Streamlit app
st.set_page_config(page_title="Cognitive Query System")

# Main title
st.title("Cognitive Query System Using Generative AI 🤖")

# Sidebar menu
st.sidebar.title("Menu")
st.sidebar.subheader("Select Query Type")

# Main options for query types
option = st.sidebar.radio("Choose an option:", ["Ask a Question", "Analyze an Image", "Explore a Document"])

# Define functions for each query type
def text_query():
    st.header("Ask Your Queries Using CQS 💬")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
    
    # Input field for new messages
    input_text = st.text_input("Input Prompt: ", key="input_text")
    
    if st.button("Submit"):
        if input_text:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": input_text})
            
            # Get AI response
            response = get_gemini_response(input_text)
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to refresh the chat display
            st.rerun()
    
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def image_query():
    st.header("Chat with Image Using CQS 🖼️")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = None
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")

    input_text = st.text_input("Input Prompt: ", key="input_image")
    
    if st.button("Submit"):
        if image:
            with st.spinner("Analyzing image..."):
                try:
                    response = get_gemini_response_image(input_text, image)
                    st.subheader("The Response is:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
        else:
            st.warning("Please upload an image before submitting your query.")

def document_query():
    st.header("Chat with Documents Using CQS 📄")

    # Dropdown menu for document type selection
    doc_type = st.selectbox("Select Document Type", ["Select...", "PDF Document", "Word Document", "Spreadsheet Document"])

    # PDF Section
    if doc_type == "PDF Document":
        st.subheader("PDF Documents")
        pdf_docs = st.file_uploader("Upload PDF Files (.pdf)", type=["pdf"], key="pdf_uploader", accept_multiple_files=True)
        user_question_pdf = st.text_input("Input Prompt: ", key="pdf_question")
        if st.button("Submit"):
            if user_question_pdf and pdf_docs:
                with st.spinner("Processing PDF..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text:
                            st.error("Could not extract text from the PDF(s). Please check the file(s) and try again.")
                            return
                        
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Failed to process the text content.")
                            return
                        
                        if not get_vector_store(text_chunks):
                            st.error("Failed to create vector store.")
                            return
                        
                        response = user_input(user_question_pdf)
                        st.subheader("The Response is:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
            else:
                st.warning("Please upload PDF files and enter a question.")

    # Word Document Section
    elif doc_type == "Word Document":
        st.subheader("Word Documents")
        uploaded_file = st.file_uploader("Choose a Word document...", type=["docx"], key="word_uploader")
        document_text = ""
        
        if uploaded_file is not None:
            # Extract text from the uploaded Word document
            document_text = extract_text_from_docx(uploaded_file)
            st.text_area("Document Content:", value=document_text, height=300)
        
        input_prompt = st.text_input("Input Prompt:", key="input_word")
        
        if st.button("Submit"):
            if document_text and input_prompt:
                with st.spinner("Analyzing document..."):
                    try:
                        response = get_gemini_response_word(input_prompt, document_text)
                        st.subheader("The Response is:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error analyzing document: {str(e)}")
            else:
                st.warning("Please upload a Word document and enter a question.")

    # Spreadsheet Document Section
    elif doc_type == "Spreadsheet Document":
        st.subheader("Spreadsheet Documents")
        uploaded_file = st.file_uploader("Choose a spreadsheet file...", type=["xlsx", "csv"], key="spreadsheet_uploader")
        spreadsheet_df = None
        
        if uploaded_file is not None:
            # Read the uploaded spreadsheet
            spreadsheet_df = read_spreadsheet(uploaded_file)
            if spreadsheet_df is not None:
                st.dataframe(spreadsheet_df.head(10))
        
        input_prompt = st.text_input("Input Prompt:", key="input_spreadsheet")
        
        if st.button("Submit"):
            if spreadsheet_df is not None and input_prompt:
                with st.spinner("Analyzing spreadsheet..."):
                    try:
                        response = get_gemini_response_spreadsheet(input_prompt, spreadsheet_df)
                        st.subheader("The Response is:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error analyzing spreadsheet: {str(e)}")
            else:
                st.warning("Please upload a spreadsheet and enter a question.")

# Conditional rendering based on the selected option
if option == "Ask a Question":
    text_query()
elif option == "Analyze an Image":
    image_query()
elif option == "Explore a Document":
    document_query()