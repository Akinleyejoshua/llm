from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prompt import prompt_gpt

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/api/gpt/prompt")
def root(prompt: str):
    output = prompt_gpt(prompt)
    return {
        "text": f"{output}"
    }