from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import JSONResponse

from src.backend.core.services import RAGService
from src.backend.infrastructure.controllers.dependencies import rag_service
from src.backend.infrastructure.controllers.schemas import QuestionRequest

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post("/generate_answer")
async def generate_answer(
    question_request: QuestionRequest,
    rag_service: Annotated[RAGService, Depends(rag_service)],
):
    try:
        query: str = question_request.query
        answer = await rag_service.generate_answer(query)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generation: {e}.")

    return JSONResponse(content={"answer": answer})
