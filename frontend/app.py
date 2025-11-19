# frontend/app.py

import streamlit as st
import requests
import time
import os

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- UI Setup ---
st.set_page_config(page_title="Smart Knowledge Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Smart Knowledge Assistant")
st.write("Upload your documents, ask questions, and get intelligent answers.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header("Add to Knowledge Base")

    # PDF Uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{BACKEND_URL}/api/documents/upload", files=files)
                    if response.status_code == 202:
                        st.success(f"Successfully uploaded '{uploaded_file.name}'.")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    st.divider()

    # YouTube URL
    youtube_url = st.text_input("Enter a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Process YouTube Video"):
        if youtube_url:
            with st.spinner("Processing YouTube URL..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/api/documents/youtube", json={"url": youtube_url})
                    if response.status_code == 202:
                        st.success("Successfully submitted YouTube URL.")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please enter a YouTube URL.")

    st.divider()

    # Web Page URL
    web_url = st.text_input("Enter a Web Page URL", placeholder="https://example.com/article")
    if st.button("Process Web Page"):
        if web_url:
            with st.spinner("Processing Web Page URL..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/api/documents/web", json={"url": web_url})
                    if response.status_code == 202:
                        st.success("Successfully submitted Web Page URL.")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please enter a web page URL.")

    st.divider()
    
    # --- Document Management Section ---
    st.header("Manage Documents")
    
    # Function to fetch documents
    def fetch_documents():
        try:
            response = requests.get(f"{BACKEND_URL}/api/documents")
            if response.status_code == 200:
                return response.json()
            else:
                st.error("Failed to fetch documents.")
                return []
        except Exception as e:
            st.error(f"Connection error: {e}")
            return []

    # Function to delete document
    def delete_document(doc_id, filename):
        try:
            response = requests.delete(f"{BACKEND_URL}/api/documents/{doc_id}")
            if response.status_code == 204:
                st.success(f"Deleted '{filename}'")
                time.sleep(1) # Give time for success message
                st.rerun()
            else:
                st.error(f"Failed to delete: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    # Display Documents
    documents = fetch_documents()
    if documents:
        for doc in documents:
            with st.expander(f"{doc['filename']} ({doc['status']})"):
                st.caption(f"Type: {doc['source_type']}")
                st.caption(f"Uploaded: {doc['upload_date'][:10]}")
                st.caption(f"Chunks: {doc['chunk_count']}")
                if st.button("Delete", key=doc['id']):
                    delete_document(doc['id'], doc['filename'])
    else:
        st.info("No documents found.")

# --- Main Chat Interface ---
st.header("💬 Chat with Your Documents")

# Add the toggle for AI enhancement
enhance_with_ai = st.toggle("Enhance with AI's general knowledge", value=False, help="When enabled, the AI will use its own knowledge to provide more complete answers, in addition to the content from your documents.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new user query
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and Stream Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            payload = {"query": prompt, "enhance_with_ai": enhance_with_ai}
            with requests.post(f"{BACKEND_URL}/api/query", json=payload, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        full_response += decoded_chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
            message_placeholder.markdown(full_response)
        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to the backend: {e}"
            message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"An unexpected error occurred: {e}"
            message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})