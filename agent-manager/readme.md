# Agent-manager

The **Agent-manager** is a Python-based project that implements a multi-agent system for processing user queries and images. It leverages various AI agents (web, code, image, and manager agents) to handle tasks like web searches, code execution, and image analysis, coordinated through a Gradio web interface.

## Features
- **Multi-Agent System**: Includes specialized agents for web searches, code execution, image processing, and task management.
- **Gradio Interface**: A user-friendly web interface for submitting queries and uploading images.
- **Caching**: Implements a TTL cache to optimize repeated search queries.
- **Logging**: Comprehensive logging for debugging and monitoring.
- **Extensible Tools**: Supports tools like DuckDuckGo search, Wikipedia search, Python interpreter, and webpage visits.

## Requirements
- Python 3.8+
- Libraries:
  - `gradio`
  - `cachetools`
  - `pillow`
  - `smolagents`
  - `litellm`
Additionally, you need A local language model server running at `http://localhost:1234/v1` (e.g., using LM Studio or similar).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/borisyich/trains.git
   cd agent-manager
   ```
2. Ensure the `System Prompts for Agents.markdown` file is present in the project directory with appropriate prompts for the agents.
3. Start a local language model server (e.g., LM Studio) at `http://localhost:1234/v1`.

## Usage
1. Run the application:
   ```bash
   python agent-manager.py
   ```
2. Open the provided Gradio URL (e.g., `http://127.0.0.1:7860`) in your browser.
3. Enter a query in the text box and optionally upload an image.
4. Click "Submit Query" to receive a response from the agent system.

## Project Structure
- **`agent-manager.py`**: Main script containing the agent configurations, query processing, and Gradio interface.
- **`System Prompts for Agents.markdown`**: File with system prompts for the agents (web, code, image, and manager).
- **`agent_manager.log`**: Log file for debugging and tracking agent activities.

## Example Queries
- **Text Query**: "Find information about neural networks and write code for training one."
- **Image Query**: Upload an image and ask, "Describe this image and its contents."

## Limitations
- Requires a running local language model server.
- All the llm for agents is chosen like small-weighted to image only sample of agents's structure and author's skills to build system such as in example.
- Image processing is basic and can be extended for more complex tasks.
- The `api_key` in the code is a placeholder and should be replaced with a valid key if required by the language model server.
