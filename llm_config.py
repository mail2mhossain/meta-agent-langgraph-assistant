from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


open_ai_llm = ChatOpenAI(
    model_name="gpt-5",
    openai_api_key="",
    # temperature=0,
)


google_llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    google_api_key = "****",
    # temperature=0,
)

def get_llm(model_name: str) -> ChatOpenAI | ChatGoogleGenerativeAI:
    if model_name == "openai":
        return open_ai_llm
    elif model_name == "google":
        return google_llm
    elif model_name == "qwen":
        return qwen_llm
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
