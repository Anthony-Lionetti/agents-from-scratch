from groq import Groq
import dotenv
import os
import re

dotenv.load_dotenv()

def llm_call(prompt: str, system_prompt: str = "", model="meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
        model (str, optional): The model to use for the call. Defaults to "meta-llama/llama-4-scout-17b-16e-instruct".

    Returns:
        str: The response from the language model.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = [
        {"role": "system", "content":system_prompt},
        {"role": "user", "content": prompt}
        ]
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content

def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses 

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""