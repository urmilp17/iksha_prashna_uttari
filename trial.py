from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import RAG as r

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        # Process the question
        output = r.user_input(request.question)
        return {"answer": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
