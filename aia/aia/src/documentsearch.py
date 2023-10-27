from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def update_embeddings(embeddings_text, embeddings_state):
    if embeddings_text:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(embeddings_text)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        return docsearch


# Pertains to question answering functionality
def update_use_embeddings(widget, state):
    if widget:
        state = widget
        return state
