from fastapi import FastAPI, Depends, Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime, os, warnings

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

warnings.filterwarnings("ignore")
load_dotenv()

# ENV & DB INIT
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# MongoDB connection with Docker support
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
client = MongoClient(MONGODB_URL)
db = client['yt_rag']
users_collection = db['users']
session_memory_collection = db['session_memory']
global_memory_collection = db['global_memory']

# Security scheme for Swagger and auth
security = HTTPBearer()

# MODELS
class UserIn(BaseModel):
    email: str
    password: str

class QueryPayload(BaseModel):
    video_id: str
    query: str

# AUTH HELPERS
def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get("email")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token: missing email")
        return email
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# MEMORY HELPERS
def get_session_memory(email):
    mem = session_memory_collection.find_one({"email": email})
    return mem["messages"] if mem else []

def save_session_memory(email, messages):
    session_memory_collection.update_one(
        {"email": email},
        {"$set": {"messages": messages}},
        upsert=True
    )

def get_global_memory(email):
    record = global_memory_collection.find_one({"email": email})
    return record["messages"] if record else []

def update_global_memory(email, query, response):
    global_memory_collection.update_one(
        {"email": email},
        {"$push": {"messages": {"$each": [
            {"role": "user", "content": query},
            {"role": "ai", "content": response}
        ]}}},
        upsert=True
    )

# VIDEO HELPER
def extract_video_info(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        transcript = ""

    ydl_opts = {'quiet': True, 'skip_download': True, 'forcejson': True, 'noplaylist': True, 'extract_flat': False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        'transcript': transcript,
        'video_title': info['title'],
        'uploader_name': info['uploader'],
        'description': info['description'],
        'url': url,
        'video_id': video_id
    }

# MODEL INIT
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# FASTAPI INIT
app = FastAPI()

@app.get("/")
def welcome():
    return {"details": "YouTube RAG API"}

@app.post("/signup")
def signup(user: UserIn):
    if users_collection.find_one({"email": user.email}):
        return {"error": "Email already registered"}
    hashed_pw = pwd_context.hash(user.password)
    users_collection.insert_one({"email": user.email, "password": hashed_pw})
    return {"msg": "Signup successful"}

@app.post("/login")
def login(user: UserIn):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        return {"error": "Invalid credentials"}

    payload = {
        "email": user.email, 
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=6)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"token": token}

@app.post("/ask")
def ask_video_question(payload: QueryPayload, email: str = Depends(get_current_user)):
    info = extract_video_info(payload.video_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([info['transcript']])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mqr)

    def get_context(inputs):
        docs = compression_retriever.invoke(inputs["question"])
        return "\n\n".join(doc.page_content for doc in docs)

    chat_history = get_session_memory(email)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant answering questions from a YouTube video.\n\n"
         "Video Title: {video_title}\nChannel: {channel_name}\nURL: https://youtube.com/watch?v={video_id}\n\n"
         "Description: {description}\n\nUse only this transcript context to answer the question:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    retrieval_chain = RunnableParallel({
        "context": RunnableLambda(get_context),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    })

    chain = retrieval_chain | prompt.partial(
        video_title=info["video_title"],
        channel_name=info["uploader_name"],
        video_id=info["video_id"],
        description=info["description"]
    ) | llm | StrOutputParser()

    response = chain.invoke({"question": payload.query, "chat_history": chat_history})

    chat_history += [
        {"role": "user", "content": payload.query},
        {"role": "ai", "content": response}
    ]
    save_session_memory(email, chat_history)
    update_global_memory(email, payload.query, response)

    return {"response": response}

@app.post("/clear-session")
def clear_session(email: str = Depends(get_current_user)):
    session_memory_collection.delete_one({"email": email})
    return {"msg": "Session memory cleared"}

@app.post("/clear-global")
def clear_global(email: str = Depends(get_current_user)):
    global_memory_collection.delete_one({"email": email})
    return {"msg": "Global memory cleared"}

@app.get("/verify-token")
def verify_token(email: str = Depends(get_current_user)):
    return {"email": email, "message": "Token is valid"}