import streamlit as st
from transformers import pipeline, set_seed
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.subheader('Generative AI')

t1, t2 = st.tabs(['Generate Text', 'AI Translator'])

with t1:
    initial_text = st.text_input('Initial text')

    if st.button('Submit'):
        generator = pipeline('text-generation', model='gpt2-large')
        set_seed(42)
        result = generator(initial_text, max_length=30, num_return_sequences=5)
        st.write(result)

with t2:
    language1 = st.text_input('Select 1st language')
    language2 = st.text_input('Select 2nd language')
    text_for_translate = st.text_input('Text to be translated')
    input_text = f'Translate {language1} to {language2}: {text_for_translate}'


    if st.button('Translate'):
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

        # input_text = "translate English to German: What are you doing?"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        outputs = model.generate(input_ids)

        st.write(tokenizer.decode(outputs[0]))

