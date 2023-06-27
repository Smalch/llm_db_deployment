from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from chain import Chain
import uvicorn
path = './Training_materials/'
qa_chain = Chain()
app = FastAPI()
answer = None

class Query(BaseModel):
    query: str


@app.post("/last_answer")
async def last_answer():
    global answer
    return answer



@app.post("/qa_from_files")
async def qa_from_files(query: Query):
    global answer
    answer = None
    answer = qa_chain.ask(query.query)
    return answer

@app.post('/upload_pdf')
async def upload_pdf(file: UploadFile = File(...)):
    with open(path+file.filename, "wb") as buffer:
        buffer.write(file.file.read())
    qa_chain.add_document(path+file.filename)
    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="http://localhost", port=8000)