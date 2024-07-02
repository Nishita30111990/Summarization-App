# streamlit - web application
# Process for application development
# create a virtual enviornment in conda
# $conda create -n summarization_jun23
# $conda activate summarization_jun23
# $conda install pip
# $pip install streamlit langchain_community langchain pandas openai
# install the required libraries
# streamlit, langchain_community, langchain, pandas
# langchain is a frame work that is used for application developemnt using LLMs. they contain agents, chains and tools
# required libraries

import streamlit as st
st.title("Summarization Application")
text_content = st.text_area("Paste the text content here:", height=150)

# $streamlit run summarization_june23.py

## Langchain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage #prompt template
import os

# Prompting template
# System Message: is a task that you give llm to perform.
# example: Summarize the below content in one sentence
# Human Message: pass the actual text data to summarize
# example: text data that u summarize
# AI Message: that LLM will deliver/output
# example: summarized text
# please login to the website https://platform.openai.com/docs/overview
# 

# Azure - Openai, OpenAI
# deploy a model - deployment name, api_key, version
# api_key: 15 requests per minute
# application: create one key - no of tokens your are passing to openai
# 1000 tokens - api - $0.002
# 500 tokens - api - $0.001
# 500 tokens - api - $0.001

os.environ["OPENAI_API_KEY"] = "sk-proj-t1Vefz76tONF6t555nb8T3BlbkFJ89TFF81Z3ONo4DTdRuv8"

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

# LLM - Prompt
# Second aspect: LLM Paramters
# Temperature: lower the temperature [close to 0], more deterministic the results
# Higher the temperature [greater than 0.5], more creative and diverse outputs

# Top P: A sampling technique with temperature, 
# low Top P: exact, factual responses, keep the value top P as less
# high Top P:  will enable the model to look for more possible words, including the less likely ones, leading to more diverse outputs

# general recommendation is to alter temperature or Top P but not both

# less temperature and high top_p [lesser confident]
# LLM - A sampling technique ?? [with [0.9], of[0.89], had[0.4], has[0.1], ....]


# temerature - 0 : relative probablities of all tokens, making it to a more global adjustment
# Top_P: 1 [more possible words, but not the exact] - accuracy:
# more targeted adjustment, dynamically changing the pool of candidate tokens

# max_length: context of LLM where it controls the total number of tokens: input+output
# max_input_length = input prompt
# max_output_length = output

# frequency penalty: applies a penalty on the next token proportinal to how many times a token already appeared
# higher frequency penalty: less likely the word appears
# lower frequency penalty: more likely the word appears