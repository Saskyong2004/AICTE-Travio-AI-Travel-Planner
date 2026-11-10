"""
llm_agent.py
============
LLM wrapper via ibm-watsonx-ai SDK (IBM watsonx.ai).
Provides generate_text() and chat_with_agent() used by the Streamlit UI.
The active model is read from IBM_MODEL_ID in .env; no silent fallback is used
for the model choice — only the env default below applies when the key is absent.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Default model — used only when IBM_MODEL_ID is not set in .env.
# Change this constant OR set IBM_MODEL_ID in .env to switch models.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "ibm/granite-4-h-small"


def get_active_model_id() -> str:
    """Return the model ID that will be used for all inference calls."""
    return os.getenv("IBM_MODEL_ID", DEFAULT_MODEL_ID)


def get_model_label() -> str:
    """
    Return a short human-readable label for the active model.
    Used by the UI to display branding text.
    """
    mid = get_active_model_id().lower()
    if "granite" in mid:
        return "IBM Granite"
    if "llama" in mid:
        # e.g. "meta-llama/llama-3-3-70b-instruct" -> "Llama 3.3"
        parts = mid.split("/")[-1].split("-")
        # Find version segments: llama-3-3-... -> "3.3"
        nums = []
        for p in parts[1:]:
            if p.isdigit():
                nums.append(p)
            else:
                break
        version = ".".join(nums) if nums else ""
        return f"Llama {version}".strip()
    # Fallback: use the part after the last "/"
    return get_active_model_id().split("/")[-1]


def get_model():
    """
    Return an ibm-watsonx-ai ModelInference instance for the configured model.
    Raises ImportError / ValueError with a helpful message when credentials
    are not configured.
    """
    api_key    = os.getenv("IBM_API_KEY", "")
    project_id = os.getenv("IBM_PROJECT_ID", "")
    model_id   = get_active_model_id()
    ibm_url    = os.getenv("IBM_URL", "https://au-syd.ml.cloud.ibm.com")

    if not api_key or api_key == "your_ibm_api_key_here":
        raise ValueError(
            "IBM_API_KEY is not set. Please update your .env file with valid credentials."
        )
    if not project_id or project_id == "your_ibm_project_id_here":
        raise ValueError(
            "IBM_PROJECT_ID is not set. Please update your .env file with your watsonx project ID."
        )

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        credentials = Credentials(url=ibm_url, api_key=api_key)

        # Parameters valid for both Granite and Llama 3.x on watsonx.ai.
        # "greedy" decoding is deterministic and does not require temperature/top_k/top_p.
        params = {
            GenParams.DECODING_METHOD:    "greedy",
            GenParams.MAX_NEW_TOKENS:     2048,
            GenParams.MIN_NEW_TOKENS:     1,
            GenParams.REPETITION_PENALTY: 1.05,
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
    Send a prompt to the configured model via watsonx.ai and return the generated text.
    Returns a human-readable error string on failure so the UI can display it.
    """
    full_prompt = prompt
    if system_context:
        full_prompt = f"{system_context}\n\n{prompt}"

    try:
        model = get_model()
        response = model.generate_text(prompt=full_prompt)
        return response
    except ValueError as exc:
        return f"⚠️ Configuration Error: {exc}"
    except ImportError as exc:
        return f"⚠️ Package Error: {exc}"
    except Exception as exc:
        # Surface the full error message so it is visible in the UI
        return f"⚠️ Model Error ({type(exc).__name__}): {exc}"


def chat_with_agent(user_message: str, conversation_history: list) -> str:
    """
    Multi-turn chat wrapper.
    conversation_history: list of {"role": "user"|"assistant", "content": str}

    Uses a plain-text prompt format compatible with both Granite and Llama models
    via the watsonx.ai text-generation endpoint.
    """
    system_prompt = (
        "You are an expert AI Travel Planner. "
        "You help users plan trips with personalized itineraries, destination recommendations, "
        "transport options, accommodation suggestions, packing tips, and travel advice. "
        "Be friendly, specific, and practical. Always consider safety, budget, and travel logistics. "
        "When suggesting places, include brief reasons why they match the user's preferences."
    )

    # Build conversational context (last 3 exchanges to stay within token limits)
    history_text = ""
    for turn in conversation_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation so far:\n{history_text}"
        f"User: {user_message}\n"
        f"Assistant:"
    )

    return generate_text(full_prompt)
