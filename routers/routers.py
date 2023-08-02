from fastapi import APIRouter
from fastapi import File, UploadFile
from services.services import transcribe


router = APIRouter()


@router.get('/')
def index():
    return {'message': 'Hello from ASR'}

@router.post('/transcription') #/transcribe
async def asr(wav: UploadFile = File(...), device: str = "cpu"):
    return transcribe(wav)