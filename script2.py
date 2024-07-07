# Step 1: Import All the Required Libraries

from googletrans import Translator
translator= Translator()

# We are creating a Webapp with Streamlit
import streamlit as st

# Replicate is an online cloud platform that allows us to host models and access the models through API
#Llama 2 models with 7B, 13 B and with 70B parameters are hosted on Replicated and we will access these models through API


import replicate
import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata # 'filter_complex_metadata' removes complex metadata that are not in str, int, float or bool format

pdf_loader = UnstructuredPDFLoader("dataset.vi.en.pdf")
pdf_doc = pdf_loader.load()
updated_pdf_doc = filter_complex_metadata(pdf_doc)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunked_pdf_doc = text_splitter.split_documents(updated_pdf_doc)
len(chunked_pdf_doc)

# Mở tệp văn bản a.txt để đọc
# with open('chunks.txt', 'r') as file:
#     chunked_pdf_doc = file.read()

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

from langchain_community.vectorstores import Chroma
db_pdf = Chroma.from_documents(chunked_pdf_doc, embeddings)

from textwrap import fill
from langchain.prompts import PromptTemplate

# Step 2: Add a title to your Streamlit Application on Browser

st.set_page_config(page_title="🦙💬 Llama 2 Chatbot with Streamlit")


#Create a Side bar
with st.sidebar:
    st.title("🦙💬 Llama 2 Chatbot")
    st.header("Settings")

    add_replicate_api=st.text_input('Enter the Replicate API token', type='password')
    if not (add_replicate_api.startswith('r8_') and len(add_replicate_api)==40):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

    st.subheader("Models and Parameters")

    select_model=st.selectbox("Choose a Llama 2 Model", ['Llama 2 7b', 'Llama 2 13b', 'Llama 2 70b'], key='select_model')
    if select_model=='Llama 2 7b':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif select_model=='Llama 2 13b':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    else:
        llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

    temperature=st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p=st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length=st.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

    st.markdown('I make content on AI on regular basis do check my Youtube channel [link](https://www.youtube.com/@muhammadmoinfaisal/videos)')

os.environ['REPLICATE_API_TOKEN']=add_replicate_api

#Store the LLM Generated Reponese

if "messages" not in st.session_state.keys():
    st.session_state.messages=[{"role": "assistant", "content":"Chào em, tôi là trợ lý ảo được tạo ra bởi HUS, chuyên giải đáp những thắc mắc về quy chế đào tạo, công tác sinh viên. Em có câu hỏi gì không ?"}]

# Diplay the chat messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Clear the Chat Messages
def clear_chat_history():
    st.session_state.messages=[{"role":"assistant", "content": "Chào em, tôi là trợ lý ảo được tạo ra bởi HUS, chuyên giải đáp những thắc mắc về quy chế đào tạo, công tác sinh viên. Em có câu hỏi gì không ?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Create a Function to generate the Llama 2 Response
def generate_llama2_response(prompt_input):
    prompt_input = translator.translate(prompt_input, dest='en').text
    default_system_prompt="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Use the following context to Answer the question at the end. Do not use any other information. If you can't find the relevant information in the context, just say you don't have enough information to answer the question, or say 'go to the training department to resolve it'. Don't try to make up an answer."
    for data in st.session_state.messages:
        print("Data:", data)
        if data["role"]=="user":
            default_system_prompt+="User: " + data["content"] + "\n\n"
        else:
            default_system_prompt+="Assistant" + data["content"] + "\n\n"
    output=replicate.run(llm, input={"prompt": f"{default_system_prompt} {prompt_input} Assistant: ",
                                     "temperature": temperature, "top_p":top_p, "max_length": max_length, "repititon_penalty":1})
    # print("Output:", output)

    return output


#User -Provided Prompt

if prompt := st.chat_input(disabled=not add_replicate_api):
    st.session_state.messages.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a New Response if the last message is not from the asssistant

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response=generate_llama2_response(prompt)
            placeholder=st.empty()
            full_response=''
            for item in response:
                full_response+=item
                placeholder.markdown(full_response)
            full_response = translator.translate(full_response, dest='vi').text
            placeholder.markdown(full_response)

    message= {"role":"assistant", "content":full_response}
    st.session_state.messages.append(message)
    # st.write(message)