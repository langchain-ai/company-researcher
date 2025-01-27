from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.graph import get_agent_graph
import uvicorn

load_dotenv()

app = FastAPI(port=8000)


class ResearchRequest(BaseModel):
    company: str
    user_notes: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/research")
async def research_company(request: ResearchRequest):
    agent = get_agent_graph()
    result = await agent.ainvoke(
        {
            "company": request.company,
            "user_notes": request.user_notes,
        }
    )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
