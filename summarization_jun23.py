import streamlit as st
st.title("Summarization Application")
text_content = st.text_area("Paste the text content here:", height=150)

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage #prompt template
import os

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                temperature=0, 
                max_tokens=500)

from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

def summarize_pdf(pdf_file_path, custom_prompt=''):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm,chain_type='map_reduce')
    summary =chain.run(docs)
    return summary


uploaded_file = st.file_uploader("Choose a file")

if st.button("summarize"):
    if uploaded_file is not None:
        summary = summarize_pdf(uploaded_file)
        st.write(summary)


    if text_content:
        summary_prompt = [
            SystemMessage(content="Summarise the given content in a sentence. Provide citations based on the paragraphs whenever possible: \n"),
            HumanMessage(content=text_content)
            ]
        output = llm(summary_prompt).content
        # display the output
        st.write("Summary:")
        st.write(output)


else:
    st.warning("Please paste the text content to proceed further")
