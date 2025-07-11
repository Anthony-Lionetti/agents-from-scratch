from llm import InvokationCore, OutputExtractor
from groq import Groq
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

def main():
    llm_provider = Groq(api_key=os.getenv("GROQ_API_KEY"))
    llm = InvokationCore(provider=llm_provider, extractor=OutputExtractor())


    system_prompt = "You are an honset, helpful and harmless priate. Respond to the users query in the most humorous, yet helpful way possible"
    prompt = "I want to get into sailing. I want to know how I can start learning how to sailboat."

    result = llm.invoke(prompt=prompt, system_prompt=system_prompt)
    
    print("Writing Results to markdown")
    Path("output/").mkdir(exist_ok=True)
    with open("output/results.md", "w+") as f:
        f.write(result)

    

if __name__ == "__main__":
    # main()

    from prompts.util import format_md_prompt
    import datetime as dt
    date = dt.datetime.now().strftime("%m/%d/%Y")

    path = "prompts/research_lead.md"
    prompt = format_md_prompt(path=path, current_date=date)
    print(prompt)