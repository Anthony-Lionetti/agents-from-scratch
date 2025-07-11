
def format_md_prompt(path:str, **kwargs) -> str:
    """
    General purpose function that reads a markdown file and flexibly injects variables into the template 
    """
    with open(path, "r") as f:
        prompt = f.read()
    
    return prompt.format(**kwargs)
    
