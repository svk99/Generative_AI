import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from io import StringIO
import pandas as pd

HUGGINGFACEHUB_API_TOKEN = ''

st.title('MCQ Generator')

TEXT = None
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()
    TEXT=string_data

NUMBER = st.number_input('Select number of question',min_value=1,max_value=5,step=1)
DIFFICULTY = st.selectbox('Select difficulty level',('Easy','Medium','Hard'))

TEMPLATE="""
You are master in generating multiple choice question.
Create {number} mcq questions with {difficulty} difficulty for the given text below.
Text:{topic}
"""

mcq_prompt=PromptTemplate(
    input_variables=['topic','number','difficulty'],
    template=TEMPLATE
    )

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

if st.button('Generate'):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    llm_chain = mcq_prompt | llm
    result = llm_chain.invoke({'topic':TEXT,'number':NUMBER,'difficulty':DIFFICULTY})
    st.write(result)