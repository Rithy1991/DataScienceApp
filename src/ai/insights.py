from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class InsightResult:
    summary: str
    warnings: str
    recommended_actions: str


def local_slm_available() -> bool:
    """Check if local transformers SLM (text2text-generation) is available."""
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def _local_transformers_available() -> bool:
    """Deprecated: use local_slm_available() instead."""
    return local_slm_available()


def _openai_compatible_available() -> bool:
    try:
        import requests  # noqa: F401

        return True
    except Exception:
        return False


def get_local_slm_pipeline():
    """Get the local SLM pipeline (text2text-generation) from Hugging Face."""
    if not local_slm_available():
        raise RuntimeError("transformers not installed. Install with: pip install transformers")
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base")


def explain_insight_simple(text: str) -> str:
    """Generate a simple insight explanation using local SLM."""
    if not local_slm_available():
        return f"[SLM unavailable] {text[:100]}"
    try:
        pipe = get_local_slm_pipeline()
        result = pipe(f"Explain this data insight concisely: {text}", max_length=100)
        return result[0].get("generated_text", text)
    except Exception as e:
        return f"Error generating insight: {e}"


def _make_prompt(df: pd.DataFrame, context: str) -> str:
    head = df.head(25).to_csv(index=False)
    stats = df.describe(include="all").fillna("").to_string()
    return (
        "You are DataScope Pro, an expert Senior Data Scientist and Analyst."
        "Your goal is to provide a high-value, executive-level summary of the provided dataset.\n"
        "Instructions:\n"
        "- Analyze the provided data stats and preview carefully.\n"
        "- Do NOT hallucinate data. If the answer is not in the stats, state that.\n"
        "- Use professional, concise language. Avoid fluff.\n"
        "- Focus on actionable business insights, not just describing the columns.\n\n"
        f"User Specific Focus: {context}\n\n"
        "Data Context:\n"
        f"Preview (First 25 rows):\n{head}\n\n"
        f"Statistics:\n{stats}\n\n"
        "Response Format:\n"
        "## 1. Executive Summary\n"
        "[Overview of the data landscape]\n\n"
        "## 2. Key Trends & Patterns\n"
        "- [Bullet point]\n"
        "- [Bullet point]\n\n"
        "## 3. Risks & Anomalies\n"
        "[Identify outliers, missing data risks, or suspicious patterns]\n\n"
        "## 4. Strategic Recommendations\n"
        "[Next steps for modeling or business action]"
    )


def generate_insights(
    df: pd.DataFrame,
    provider: str,
    model: str,
    max_new_tokens: int = 256,
    context: str = "",
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> InsightResult:
    prompt = _make_prompt(df, context=context)

    if provider == "openai_compatible":
        if not _openai_compatible_available():
            raise RuntimeError("requests is required for openai_compatible provider")
        import requests

        base = openai_base_url or os.getenv("OPENAI_BASE_URL")
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not base or not key:
            raise RuntimeError("Missing OPENAI_BASE_URL/OPENAI_API_KEY (set in environment or Streamlit secrets)")

        # Minimal OpenAI-compatible chat completion request
        url = base.rstrip("/") + "/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": int(max_new_tokens),
        }
        resp = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _postprocess(content)

    # default: local small model via transformers
    if not _local_transformers_available():
        raise RuntimeError("Local SLM requires 'transformers' + 'torch'. Install optional AI dependencies.")

    from transformers import pipeline

    gen = pipeline("text2text-generation", model=model)
    out = gen(prompt, max_new_tokens=int(max_new_tokens), do_sample=False)
    text = out[0]["generated_text"]
    return _postprocess(text)


def _postprocess(text: str) -> InsightResult:
    text = (text or "").strip()

    # Simple section extraction; if missing, return whole text as summary.
    summary = text
    warnings = ""
    actions = ""

    lower = text.lower()
    if "risks" in lower or "anomal" in lower or "recommended" in lower:
        # keep as-is; UI will show full block
        summary = text

    return InsightResult(summary=summary, warnings=warnings, recommended_actions=actions)
