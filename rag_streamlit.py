import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("💬 RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show a loading spinner while bot processes
    with st.spinner("Thinking..."):
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index("integrated-dense-rag")

            # Pinecone search
            results = index.search(
                namespace="rag-integrated-embed",
                query={
                    "inputs": {"text": user_input},
                    "top_k": 3
                },
                fields=["chunk_text"]
            )

            # Build context from search hits
            context = ""
            for hit in results['result']['hits']:
                context += hit['fields']['chunk_text'] + "\n\n"

            # Construct prompt
            prompt = f"""
            CONTEXT: {context}
            QUERY: {user_input}
            """

            # Initialize Gemini/OpenAI client
            client = OpenAI(
                api_key=os.getenv('GEMINI_API_KEY'),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

            # Call chat completion API
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": "You are an assistant chatbot. Answer the question based on the context provided."},
                    {"role": "user", "content": prompt}
                ]
            )

            bot_reply = response.choices[0].message.content

        except Exception as e:
            bot_reply = f"⚠️ Error: {str(e)}"

    # Display assistant reply
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})