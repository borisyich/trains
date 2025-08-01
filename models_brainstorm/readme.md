# Models-brainstorm

This application simulates a brainstorming process using large language models (LLMs) via the Ollama API. It generates ideas, critiques them, and iteratively improves them, mimicking a generative adversarial network (GAN) approach for text-based ideation.

## Features
- **Idea Generation**: Produces initial ideas based on a user-defined prompt.
- **Critique and Improvement**: Critiques generated ideas and refines them over multiple iterations.
- **Result Saving**: Saves the entire brainstorming process, including prompts, configurations, and iteration history, in Markdown files.
- **Configurable**: Uses external text files for prompts and a JSON file for model configurations.

## Project Structure
```
├── app.py           # Main application script
├── configs/               # Directory for configuration and prompt files
│   ├── config.json        # Model configuration file
│   ├── generate_prompt.txt # Prompt for idea generation
│   ├── critique_prompt.txt # Prompt for critiquing ideas
│   ├── improve_prompt.txt  # Prompt for improving ideas
│   ├── initial_prompt.txt  # Initial brainstorming prompt
├── results/               # Directory for output Markdown files
```

## Requirements
- Python 3.7+
- Libraries: 
  - `datetime`
  - `glob`
  - `json`
  - `os`
  - `requests`
  - `time`
- Ollama API running locally (default ports: 11434 for generation/improvement, 11435 for critique)
- Model: `gemma3n:e2b` (or as specified in `config.json`)

## Installation
1. Clone the repository 
  ```powershell
   git clone https://github.com/borisyich/trains.git
   cd trains/models-brainstorm
   ```
2. **Install Ollama**: Follow the [Ollama documentation](https://ollama.ai/docs) to set up the API server.
3. Ensure Ollama is running with the Gemma3n:e2b model:
   ```powershell
   ollama run gemma3n:e2b
   ```
4. Ensure second Ollama is running with the Gemma3n:e2b model in the second window:
   ```powershell
   $env:OLLAMA_HOST="127.0.0.1:11435"; ollama run gemma3n:e2b
   ```

## Usage
1. Describe your exercise and write it in initial_promt (`./configs/initial_prompt.txt`)
2. Run the app in the third window:
   ```powershell
   python app.py
   ```
3. Check observations in new file: `./results/brainstorm_XXX.md`

## Output Format
Each output file (`results/brainstorm_XXX.md`) contains:
- Experiment metadata (date, time, execution time)
- Initial prompt
- Prompt file contents
- Configuration file content
- Iteration history (initial idea, critiques, improved ideas)
- Final idea
## Customization
- **Adjust Prompts**: Edit the `.txt` files in `configs/` to tailor the generation, critique, or improvement process.
- **Model Settings**: Update `config.json` to adjust model parameters or specify a different model.
