# Import required libraries
import gradio as gr
import lmstudio as lms
import logging

from cachetools import TTLCache
from datetime import datetime
from smolagents import (CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, FinalAnswerTool,
                        WikipediaSearchTool, PythonInterpreterTool, LiteLLMModel)

# Configure logging
logging.basicConfig(
    filename='agent_manager.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize cache with 100 entries and 1-hour TTL
search_cache = TTLCache(maxsize=100, ttl=3600)

# Read system prompts from markdown file
try:
    with open("System Prompts for Agents.markdown", 'r') as md:
        prompt = md.readlines()
        web_prompt = ''.join(prompt[35:54])
        code_prompt = ''.join(prompt[60:77])
        image_prompt = ''.join(prompt[84:])
        manage_prompt = ''.join(prompt[7:28])
        web_description = ''.join(prompt[32])[10:-1]
        code_description = ''.join(prompt[57])[10:-1]
        image_description = ''.join(prompt[81])[10:-1]
        manage_description = ''.join(prompt[4])[10:-1]
    logger.info("Successfully loaded system prompts from markdown file")
except FileNotFoundError:
    logger.error("System Prompts for Agents.markdown file not found")
    raise
except Exception as e:
    logger.error(f"Error reading system prompts: {str(e)}")
    raise

# Define API key (placeholder)
api_key = "dummy"

# Initialize web agent language model
web_llm = LiteLLMModel(
    model_id="openai/qwen/qwen3-1.7b", 
    api_base="http://localhost:1234/v1",
    api_key=api_key,
    temperature=0.7
    )
# Initialize code agent language model
code_llm = LiteLLMModel(
    model_id="openai/qwen/qwen3-1.7b", 
    api_base="http://localhost:1234/v1",
    api_key=api_key,
)
# Initialize image agent language model
image_llm = LiteLLMModel(
    model_id="openai/smolvlm2-2.2b-instruct", 
    api_base="http://localhost:1234/v1",
    api_key=api_key,
)
# Initialize manager agent language model
manage_llm = LiteLLMModel(
    model_id="openai/deepseek/deepseek-r1-0528-qwen3-8b",
    api_base="http://localhost:1234/v1",
    api_key=api_key,
    )

# Configure web agent with search tools
web_agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        WikipediaSearchTool(),
        FinalAnswerTool(),
    ],
    model=web_llm,
    additional_authorized_imports=[
        "requests",
        "json",
        "re"
    ],
    name="web_agent",
    description=web_description,
    max_steps=10,
    max_print_outputs_length=8096,
)
# Configure code agent with Python execution capabilities
code_agent = CodeAgent(
    tools=[
        PythonInterpreterTool(),
        FinalAnswerTool(),
    ],
    model=code_llm,
    additional_authorized_imports=[
        "json",
        "pandas",
        "numpy",
        "requests",
        "time",
        "datetime",
        "re"
    ],
    name="code_agent",
    description=code_description,
    planning_interval=5,
    max_steps=10,
    max_print_outputs_length=8096,
)
# Configure image agent for image-related tasks
image_agent = CodeAgent(
    tools=[
        VisitWebpageTool(),
        FinalAnswerTool(),
    ],
    model=image_llm,
    additional_authorized_imports=[
        "Pillow",
        "json",
        "requests",
    ],
    name="image_agent",
    description=image_description,
    planning_interval=5,
    max_steps=10
)
# Configure manager agent to coordinate other agents
manager_agent = CodeAgent(
    tools=[
        VisitWebpageTool(),
        PythonInterpreterTool(),
        FinalAnswerTool(),
    ],
    model=manage_llm,
    additional_authorized_imports=[
        "re",
        "json",
        "requests",
    ],
    managed_agents=[
        web_agent, 
        code_agent, 
        image_agent
        ],
    name="manager_agent",
    description=manage_description,
    planning_interval=10,
    max_steps=10,
    max_print_outputs_length=8096,
)

# Cache-enabled search function
def cached_search(query):
    """
    Perform a search with caching to avoid redundant queries.
    """
    if query in search_cache:
        logger.info(f"Cache hit for query: {query}")
        return search_cache[query]
    
    try:
        # Perform search using web agent's tools
        result = web_agent.run(query)
        search_cache[query] = result
        logger.info(f"New search result cached for query: {query}")
        return result
    except Exception as e:
        logger.error(f"Search error for query {query}: {str(e)}")
        return f"Error during search: {str(e)}"

# Process image file
def process_image(image_path):
    """
    Process an uploaded image using the image agent.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            # Simple description for demonstration; can be extended
            description = f"Image dimensions: {img.size}, Format: {img.format}"
            logger.info(f"Processed image: {image_path}")
            # Pass to image_agent for further analysis
            result = image_agent.run(f"Describe this image: {description}")
            return result
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return f"Error processing image: {str(e)}"

# Define Gradio interface function
def process_query(user_input, image=None):
    """
    Process user input through the manager agent and return the response.
    """
    try:
        logger.info(f"Received query: {user_input}")
        if image:
            logger.info(f"Image provided: {image}")
            image_result = process_image(image)
            combined_input = f"{user_input}\nImage analysis: {image_result}"
            response = manager_agent.run(combined_input)
        else:
            response = cached_search(user_input)
        logger.info(f"Query response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Error processing query: {str(e)}"

# Create Gradio interface
def create_interface():
    """
    Create and launch a Gradio web interface for interacting with the agent manager.
    """
    with gr.Blocks(title="Agent-manager Interface") as interface:
        gr.Markdown("# Agent-manager Interface")
        gr.Markdown("Enter your query and optionally upload an image to interact with the agent system.")
        
        # Input components
        query_input = gr.Textbox(
            label="Your Query",
            placeholder="E.g., Find information about neural networks and write code for training one",
            lines=3
        )
        image_input = gr.Image(type="filepath", label="Upload Image (Optional)")
        
        # Output component
        output = gr.Textbox(label="Response", lines=10)
        
        # Submit button
        submit_button = gr.Button("Submit Query")
        
        # Connect button to processing function
        submit_button.click(
            fn=process_query,
            inputs=[query_input, image_input],
            outputs=output
        )
    
    # Launch the interface
    logger.info("Launching Gradio interface")
    interface.launch()

if __name__ == "__main__":
    create_interface()