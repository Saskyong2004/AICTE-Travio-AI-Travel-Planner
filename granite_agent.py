"""
granite_agent.py  — SUPERSEDED, kept for reference only
=========================================================
This module is no longer imported by app.py.
The active LLM wrapper is llm_agent.py (uses meta-llama/llama-3-3-70b-instruct).
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _get_credentials() -> dict:
    return {
        "url": os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.getenv("IBM_API_KEY", ""),
    }


def get_model():
    """
    Return an ibm-watsonx-ai ModelInference instance for IBM Granite.
    Raises ImportError / ValueError with a helpful message when credentials
    are not configured.
    """
    api_key = os.getenv("IBM_API_KEY", "")
    project_id = os.getenv("IBM_PROJECT_ID", "")
    model_id = os.getenv("IBM_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

    if not api_key or api_key == "your_ibm_api_key_here":
        raise ValueError(
            "IBM_API_KEY is not set. Please update your .env file with valid credentials."
        )
    if not project_id or project_id == "your_ibm_project_id_here":
        raise ValueError(
            "IBM_PROJECT_ID is not set. Please update your .env file with your watsonx project ID."
        )

    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        credentials = Credentials(
            url=os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=api_key,
        )
        client = APIClient(credentials)

        params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 2048,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1,
            GenParams.REPETITION_PENALTY: 1.1,
        }

        model = ModelInference(
            model_id=model_id,
            params=params,
            credentials=credentials,
            project_id=project_id,
        )
        return model

    except ImportError:
        raise ImportError(
            "ibm-watsonx-ai package is not installed. Run: pip install ibm-watsonx-ai"
        )


def generate_text(prompt: str, system_context: Optional[str] = None) -> str:
    """
    Send a prompt to IBM Granite and return the generated text.
    Falls back to a descriptive error string on failure so the UI
    can display it gracefully.
    """
    full_prompt = prompt
    if system_context:
        full_prompt = f"{system_context}\n\n{prompt}"

    try:
        model = get_model()
        response = model.generate_text(prompt=full_prompt)
        return response
    except ValueError as e:
        return f"⚠️ Configuration Error: {e}"
    except ImportError as e:
        return f"⚠️ Package Error: {e}"
    except Exception as e:
        return f"⚠️ IBM Granite Error: {e}"


def chat_with_agent(user_message: str, conversation_history: list) -> str:
    """
    Multi-turn chat wrapper.
    conversation_history: list of {"role": "user"|"assistant", "content": str}
    """
    system_prompt = """You are an expert AI Travel Planner powered by IBM Granite.
You help users plan trips with personalized itineraries, destination recommendations,
transport options, accommodation suggestions, packing tips, and travel advice.
Be friendly, specific, and practical. Always consider safety, budget, and travel logistics.
When suggesting places, include brief reasons why they match the user's preferences."""

    # Build conversational context
    history_text = ""
    for turn in conversation_history[-6:]:  # keep last 3 exchanges for context window
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    full_prompt = f"""{system_prompt}

Conversation so far:
{history_text}
User: {user_message}
Assistant:"""

    return generate_text(full_prompt)
