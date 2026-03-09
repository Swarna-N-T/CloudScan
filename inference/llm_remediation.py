"""
LLM Remediation Module
Calls the OpenRouter API to generate remediation advice for flagged Terraform resources.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter API Setup ─────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "⚠️  Missing OPEN_ROUTER_KEY. "
        "Create a .env file in the project root with:\n"
        "  OPEN_ROUTER_KEY=your_key_here"
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Risk-level labels used in the prompt
RISK_LABELS = {0: "Safe", 1: "Low", 2: "Medium", 3: "High/Critical"}

# ── Model selection ───────────────────────────────────────────────────────────
# Default model on OpenRouter (Google Gemini 2.0 Flash via OpenRouter)
# You can change this to any model available on OpenRouter, e.g.:
#   "google/gemini-2.0-flash-001"
#   "anthropic/claude-3.5-sonnet"
#   "meta-llama/llama-3-70b-instruct"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")


def _build_prompt(flagged_resources, terraform_source=None):
    """
    Build a structured prompt for the LLM listing every flagged resource
    together with the original Terraform source file (if available).
    """
    resource_blocks = []
    for i, res in enumerate(flagged_resources, 1):
        risk_label = RISK_LABELS.get(res["predicted_risk"], "Unknown")
        block = (
            f"### Flagged Resource #{i}\n"
            f"- **Resource Type:** {res['resource_type']}\n"
            f"- **Resource ID:** {res['node_id']}\n"
            f"- **Predicted Risk Level:** {risk_label} ({res['predicted_risk']})\n"
            f"- **Configuration:**\n```json\n{json.dumps(res['config'], indent=2, default=str)}\n```"
        )
        resource_blocks.append(block)

    flagged_section = "\n\n".join(resource_blocks)

    source_section = ""
    if terraform_source:
        source_section = (
            "\n\n## Original Terraform Source\n"
            f"```hcl\n{terraform_source}\n```"
        )

    prompt = f"""You are an expert AWS cloud security engineer specializing in Terraform Infrastructure-as-Code.

An AI model (RGCN graph neural network) has analyzed a Terraform configuration and flagged the following resources as potentially misconfigured or insecure.

For EACH flagged resource below:
1. Explain what the security risk or misconfiguration is.
2. Explain the potential impact if left unaddressed.
3. Provide the corrected Terraform code block as a remediation.

{flagged_section}
{source_section}

Please provide your remediation in clear, structured format with corrected Terraform HCL code blocks."""

    return prompt


def generate_remediation(flagged_resources, terraform_source=None):
    """
    Generate remediation advice for a list of flagged Terraform resources.

    Parameters
    ----------
    flagged_resources : list[dict]
        Each dict must contain:
          - node_id       : str   (e.g. "main.tf::aws_s3_bucket.my_bucket")
          - resource_type : str   (e.g. "aws_s3_bucket")
          - predicted_risk: int   (1, 2, or 3)
          - config        : dict  (the parsed resource configuration)
    terraform_source : str, optional
        The raw .tf file contents for additional context.

    Returns
    -------
    str
        The LLM-generated remediation text.
    """
    if not flagged_resources:
        return "✅ No resources were flagged — no remediation needed."

    prompt = _build_prompt(flagged_resources, terraform_source)

    model_name = DEFAULT_MODEL
    print(f"🤖 Calling OpenRouter ({model_name}) for remediation...")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/CloudScan",   # Optional: for OpenRouter analytics
        "X-Title": "CloudScan Remediation",                # Optional: app name in OpenRouter dashboard
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AWS cloud security engineer specializing in Terraform Infrastructure-as-Code.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)

    if response.status_code != 200:
        error_detail = response.text
        raise RuntimeError(
            f"❌ OpenRouter API error (HTTP {response.status_code}): {error_detail}"
        )

    data = response.json()

    # Extract the assistant's reply
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"❌ Unexpected API response format: {e}\nFull response: {json.dumps(data, indent=2)}"
        )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test with a dummy flagged resource
    sample = [
        {
            "node_id": "main.tf::aws_s3_bucket.public_bucket",
            "resource_type": "aws_s3_bucket",
            "predicted_risk": 3,
            "config": {
                "bucket": "my-public-data",
                "acl": "public-read",
            },
        }
    ]
    result = generate_remediation(sample)
    print("\n── Remediation Output ──")
    print(result)
