from uuid import uuid4
import warnings
warnings.filterwarnings('ignore')


from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  

load_dotenv()

#constants
chunk_size=1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    yield "Initializing components...."
    initialize_components()
    yield "Resetting Vector Store...."
    vector_store.reset_collection()


    # Document Loading From Urls
    yield "Loading data...."
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    }

    loader=UnstructuredURLLoader(urls=urls,headers=headers)
    data=loader.load()
    
    # Document Splitting

    yield "Splitting data into chunks...."
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "],chunk_size=200)
    docs = text_splitter.split_documents(data)


    # Adding Documents to vector DB
    yield "Adding documents to vector database...."
    uuids=[str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    yield "Done adding docs to vector database...."




prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.

Context:
{context}

Question:
{question}
""")


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)

    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Generate answer
    answer = chain.invoke({
        "context": context,
        "question": query
    })

    # Extract sources
    sources = list(
        set(
            doc.metadata.get("source", "Unknown")
            for doc in docs
        )
    )

    return answer, sources



if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    # Consume the generator so the code inside actually executes
    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    """ results = vector_store.similarity_search(
    "30 year mortage rate",
    k=2,
    
    )
    print(results)"""
