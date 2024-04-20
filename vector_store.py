# imports
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

BOOKS_DIR = r"data\epics"
STORE_DIR = r"vectorstore"

# load pdfs from directory to documents
loader = PyPDFDirectoryLoader(path=BOOKS_DIR)
docs = loader.load() # --> list of documents
# print(len(docs)) # --> 2600

# split the text to create chunks of documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=400,
    length_function=len,
    add_start_index=True,
)

chunks = text_splitter.split_documents(docs)
# print(len(chunks)) # --> 20131    

# embed text using Hf model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", encode_kwargs = {'normalize_embeddings': True})

# create vector store from documents
faiss = FAISS.from_documents(documents=chunks, embedding=embedding_model)

# save the embeddings locally
faiss.save_local(folder_path=STORE_DIR)

