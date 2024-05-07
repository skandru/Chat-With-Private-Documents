import streamlit as st
import os
import tempfile
import base64
import gc
import random
import time
import uuid
import weaviate
from IPython.display import Markdown, display
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.settings import Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


if 'key' not in st.session_state:
    st.session_state['key'] = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state['key']
client = None

# Setup the LLM and Embedding Model
Settings.llm = Ollama(model="llama3")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

#Display the loaded file 
def display_pdf(file):
    # Opening file from file path
    st.markdown("Uploaded PDF File Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header(f"Upload your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")
                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir = temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    # Load data
                    documents = loader.load_data()
           
                    # Setup Node Parser
                    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)
                    nodes = node_parser.get_nodes_from_documents(documents)

                    # Connect to Weaviate instance
                    client = weaviate.Client(embedded_options=weaviate.embedded.EmbeddedOptions())

                    # Setup Vector Store
                    index_name = "MyExternalContext"
                    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    if client.schema.exists(index_name):
                        client.schema.delete_class(index_name)

                    index = VectorStoreIndex(nodes, storage_context=storage_context)

                    # Setup Post Processors
                    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
                    rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

                    # Setup the Query Engine
                    query_engine = index.as_query_engine(streaming=True,similarity_top_k=6, vector_store_query_mode="hybrid", alpha=0.5, node_postprocessors=[postproc, rerank])

                    # Add prompt template 
                    prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    prompt_tmpl = PromptTemplate(prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": prompt_tmpl}
                    )
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat With Uploaded Document")
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Initialize Streamlit UI
st.title('Chat With Private Documents by using Llama3')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
