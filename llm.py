from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from groq import Groq
from logger import setup_logging
import re

# My POV is that there are three different components here
# 1. How you are inferencing (InvokationClient)
# 2. How you are parsing the output (ParingClient)
# 3. How you schedule task (OrchestrationClient)

logger = setup_logging(environment="dev")


class OutputExtractor:
    def extract_xml(self, text:str, tag:str) -> str:
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
    

class InvokationClient:
    """Class that handles different patterns of calling an LLM"""
    def __init__(self, provider:Groq, extractor:OutputExtractor):
        """
        Initializes an LLM client

        Args: 
            provider (Groq): LLM inference provider 

        Note: Currently only supports Groq as an LLM provider
        """
        self.client:Groq               = provider()
        self.extractor:OutputExtractor = extractor()
    
    def invoke(self, prompt:str, system_prompt:str="", model="meta-llama/llama-4-scout-17b-16e-instruct"):
        """
        Calls the model with the given prompt and returns the response.

        Args:
            prompt (str): The user prompt to send to the model.
            system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
            model (str, optional): The model to use for the call. Defaults to "meta-llama/llama-4-scout-17b-16e-instruct".

        Returns:
            str: The response from the language model.
        """
        logger.info(f"[InvokationClient.invoke] - Invoking model: {model}")
        logger.info(f"[InvokationClient.invoke] - Prompt: {prompt}")
        messages = [
        {"role": "system", "content":system_prompt},
        {"role": "user", "content": prompt}
        ]
        logger.debug(f"[InvokationClient.invoke] - Messages: {str(messages)}")

        response = self.client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=messages,
            temperature=0.2,
        )
        response_content = response.choices[0].message.content
        logger.info(f"[InvokationClient.invoke] - Response: {response_content}")
        return response_content 
    
    def invoke_with_chain(self, input:str, prompts:List[str]) -> str:
        """
        Peform a chain of prompts using an initial input
        
        Args:
            input (str): Initial input to kick off prompt chain
            prompts (list[str]): Prompts to run sequentially in a chain on the input

        Returns:
            results (str): Final results from running the prompt chain
        """
        result = input
        for i, prompt in enumerate(prompts, 1):
            result = self.invoke(f"{prompt}\nInput: {result}")

    def invoke_with_parallel(self, prompt:str, inputs:List[str], n_workers:int = 3) -> List[str]:
        ...
    
    def invoke_with_router(self, input:str, routes: Dict[str, str]) -> str:
        """
        Route specific input to a specialized prompt using content classification.

        Args:
            input (str): Prompt to rout to get handled by a specifc prompt
            routes (dict[str,str]):\n
                Key - Route type / Category\n
                Value - Prompt detailing how to handle requests to this route
        """
        ...

    def invoke_with_eval_loop(self, task:str, generator_prompt:str, evaluator_promt:str, max_iter:int=5) -> tuple[str, list[dict]]:
        """
        Run a query until the answer passes eval requirements or hits maximum tries

        Args:
            task (str): The goal or objective to complete.
            generator_prompt (str): The prompt used to generate the answer generation
            evaluator_prompt (str): The prompt used to score the answer generation
            max_iter (int): Stop limit used to break out of loop if generated response to task does not pass
        """
        # Initialize tracking for loop
        memory, chain_of_thought = [], []

        # Generate initial thought, append results and though process to tracking
        result, thought = self.__eval_loop_generation(generator_prompt, task)
        memory.append(result)
        chain_of_thought.append(dict(thought=thought, result=result))

        # Run evaluation loop until the generation passes, or hits iteration limit
        for _ in range(max_iter):
            evaluation, feedback = self.__eval_loop_evaluation(evaluator_promt, result, task)
            if evaluation == "PASS":
                return result, chain_of_thought

            # Generate context to join previous attempts with feedback
            context = "\n".join(["Previous attempts:", *[f"- {m}" for m in memory]], f"\nFeedback: {feedback}") 

            # Generate new response from feedback
            result, thought = self.__eval_loop_generation(generator_prompt, task, context)

            # Add result and entire chain to tracking
            memory.append(result)
            chain_of_thought.append(dict(thought=thought, result=result))

        
        # Returns result and chain of though of final generation regardless of pass
        return result, chain_of_thought


    def __eval_loop_generation(self, prompt:str, task:str, context:str="") -> tuple[str, str]:
        """
        Helper function used in `invoke_with_eval_loop` method to generate a response to the task
        
        Args:
            prompt (str): prompt used to generate a response to the task
            task (str): The goal or objective to complete.
            context (str): Additional information to aid in completion of the task

        Returns:
            tuple: 
                result (str): Generated response to the task 
                thoughts (str): Thought process created before result generation
        """

        full_prompt = f"{prompt}\n{context}\nTask: {task}" if context else f"{prompt}\nTask: {task}"
        
        response = self.invoke(full_prompt)
        thoughts = self.extractor.extract_xml(response, "thoughts")
        result = self.extractor.extract_xml(response, "response")
        
        return result, thoughts

    def __eval_loop_evaluation(self, prompt:str, content:str, task:str) -> tuple[str, str]:
        """
        Helper function used in `invoke_with_eval_loop` method to generate an evaluation

        Args:
            prompt (str): prompt used to generate a response to the task
            task (str): The goal or objective to complete.
            content (str): Information to evaluate

        Returns:
            tuple: 
                evaluation (str): Generated evaluation for the generated response
                feedback (str): Details on if the generated content passed evaluation or needs edits
        """
        full_prompt = f"{prompt}\nOriginal task: {task}\nContent to evaluate: {content}"
        
        response = self.invoke(full_prompt)
        evaluation = self.extractor.extract_xml(response, "evaluation")
        feedback = self.extractor.extract_xml(response, "feedback")
        
        return evaluation, feedback

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    import os
    groq = Groq(os.getenv("GROQ_API_KEY"))
    extractor = OutputExtractor()


    llm = InvokationClient(provider=groq, extractor=extractor)