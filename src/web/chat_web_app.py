import os
import io
from uuid import uuid4
from openai import AsyncOpenAI
from pydantic import BaseModel
from chatbot.rag.cohere_rag import CohereRAG
from web.chat_web_template import IFRAME_HTML
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from chatbot.utils.paths import INDEXES_DIR, WEB_CONTENT_DIR

app = FastAPI()

USER_ID = str(uuid4())
rag = CohereRAG(
    WEB_CONTENT_DIR / "bcaitech.txt",
    INDEXES_DIR,
    chunking_type="recursive",
    rerank=True
)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API"))

class ChatRequest(BaseModel):
    query: str

def format_response(text):
    """Format the raw response text into HTML for better display"""
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('# '):
            formatted_lines.append(f'<h2>{line[2:]}</h2>')
        elif line.startswith('## '):
            formatted_lines.append(f'<h3>{line[3:]}</h3>')
        elif line.startswith('- '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{line[2:]}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(f'<p>{line}</p>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return ''.join(formatted_lines)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the iframe HTML interface"""
    return HTMLResponse(content=IFRAME_HTML)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests"""
    try:
        query = request.query.strip()
        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty query"}
            )
            
        raw_response = await rag.get_response(query, USER_ID)
        formatted_response = format_response(raw_response)
        
        return {
            "response": formatted_response,
            "user_id": USER_ID
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def transcribe(audio_buffer: io.BytesIO):
    transcription = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        response_format="text"
    )
    return transcription

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_buffer = io.BytesIO(await file.read())
        audio_buffer.name = file.filename
        transcription = await transcribe(audio_buffer)
        return {"transcription": transcription}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")

if __name__ == "__main__":
    main()
