from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import fetch_and_process_url, initialize_qa_chain
import uuid
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

# In-memory state store for user-specific data
user_state = {}


class URLInput(BaseModel):
    url: str

class QuestionInput(BaseModel):
    session_id: str
    question: str

# Endpoint to set the URL
@app.post("/set_url/")
async def set_url(data: URLInput):
    # Generate a unique session_id
    session_id = str(uuid.uuid4())

    try:
        retriever = fetch_and_process_url(data.url, session_id)
        user_state[session_id] = {"retriever": retriever}
        return {"message": "URL processed successfully. You can now ask questions.", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {e}")

# Endpoint to ask a question
@app.post("/ask_question/")
async def ask_question(data: QuestionInput):
    session_id = data.session_id
    if not session_id or session_id not in user_state:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id.")

    retriever = user_state[session_id]["retriever"]
    qa_chain = initialize_qa_chain(retriever)
    try:
        result = qa_chain.run({"query": data.question})
        print(result)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your question: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.8", port=8000, reload=True)
