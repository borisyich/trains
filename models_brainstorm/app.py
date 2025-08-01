import datetime
import glob
import json
import os
import requests
import time

# Configuration directories for storing prompt and result files
configs_dir = "./configs/"  # Directory for configuration and prompt files
results_dir = "./results/"  # Directory for saving brainstorming results

# Read configuration from a JSON file
file_path = f"{configs_dir}config.json"
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
        if not config:
            raise ValueError(f"Файл {file_path} пуст")
except FileNotFoundError:
    raise FileNotFoundError(f"Файл {file_path} не найден")
except json.JSONDecodeError:
    raise ValueError(f"Файл {file_path} содержит некорректный JSON")
except Exception as e:
    raise Exception(f"Ошибка при чтении файла {file_path}: {str(e)}")

# Define model
model_name = config["model_name"]

def read_prompt(file_path):
    """Reads a prompt from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if not content:
                raise ValueError(f"Файл {file_path} пуст")
            return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла {file_path}: {str(e)}")

def query_model(system_prompt, user_message, host="http://127.0.0.1:11434", model=model_name, options=None):
    """Sends a request to a language model via the Ollama /api/chat endpoint."""
    if options is None:
        options = {}
    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "options": options,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
    except requests.RequestException as e:
        return f"Ошибка при запросе к модели: {str(e)}"
    except KeyError:
        return "Ошибка: некорректный формат ответа от модели"

def generate_idea(prompt, host="http://127.0.0.1:11434"):
    """Generates an idea using a language model."""
    system_prompt = read_prompt(f"{configs_dir}generate_prompt.txt").format(prompt=prompt)
    options = config["generate"]
    return query_model(system_prompt, prompt, host, options=options)

def critique_idea(idea, host="http://127.0.0.1:11435"):
    """Critiques an idea and suggests improvements."""
    system_prompt = read_prompt(f"{configs_dir}critique_prompt.txt").format(idea=idea)
    options = config["critique"]
    return query_model(system_prompt, idea, host, options=options)

def improve_idea(original_idea, critique, host="http://127.0.0.1:11434"):
    """Improves an idea based on critique."""
    system_prompt = read_prompt(f"{configs_dir}improve_prompt.txt").format(idea=original_idea, critique=critique)
    user_message = f"Исходная идея: {original_idea}\nКритика: {critique}"
    options = config["improve"]
    return query_model(system_prompt, user_message, host, options=options)

def get_unique_filename(base_name="brainstorm", extension=".md"):
    """Generates a unique filename with a numeric suffix."""
    existing_files = glob.glob(f"{results_dir}{base_name}_*[0-9][0-9][0-9]{extension}")
    if not existing_files:
        return f"{base_name}_001{extension}"
    max_suffix = max(int(f.split('_')[-1].replace(extension, '')) for f in existing_files)
    return f"{base_name}_{max_suffix + 1:03d}{extension}"

def save_results(initial_prompt, history, execution_time, prompt_files, config_file=config, config_path=file_path):
    """Saves the brainstorming process and final idea to a Markdown file."""
    filename = get_unique_filename()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(f"{results_dir}{filename}", 'w', encoding='utf-8') as f:
        f.write(f"# Эксперимент: Мозговой штурм\n")
        f.write(f"**Дата и время**: {timestamp}\n")
        f.write(f"**Время выполнения**: {execution_time:.2f} секунд\n\n")
        
        f.write(f"## Начальный запрос\n{initial_prompt}\n\n")
        
        f.write(f"## Промпты\n")
        for prompt_file in prompt_files:
            prompt_content = read_prompt(f"{configs_dir}{prompt_file}")
            f.write(f"### {prompt_file}\n```plaintext\n{prompt_content}\n```\n")
        
        f.write(f"## Конфигурации\n")
        config_content = config_file
        f.write(f"### {config_path}\n```json\n{json.dumps(config_content, indent=2, ensure_ascii=False)}\n```\n")
        
        f.write(f"## Процесс итераций\n")
        f.write(f"### Начальная идея\n{history[0][0]}\n\n")
        for i, (idea, critique) in enumerate(history[1:], 1):
            f.write(f"### Итерация {i}\n")
            f.write(f"#### Критика\n{critique}\n")
            f.write(f"#### Улучшенная идея\n{idea}\n\n")
        
        f.write(f"## Финальная идея\n{history[-1][0]}\n")
    
    print(f"Результаты сохранены в {results_dir}{filename}")

def brainstorm(initial_prompt, iterations=3):
    """Orchestrates a brainstorming session between two models."""
    start_time = time.time()
    history = []
    current_idea = generate_idea(initial_prompt)
    print(f"Начальная идея:\n{current_idea}\n")
    history.append((current_idea, None)) 

    for i in range(iterations):
        print(f"Итерация {i + 1}:")
        critique = critique_idea(current_idea, host="http://127.0.0.1:11435")
        print(f"Критика:\n{critique}\n")
        current_idea = improve_idea(current_idea, critique)
        print(f"Улучшенная идея:\n{current_idea}\n")
        history.append((current_idea, critique))

    end_time = time.time()
    execution_time = end_time - start_time
    return history, execution_time

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)

    # Check for required configuration and prompt files
    required_files = [
        "generate_prompt.txt", "critique_prompt.txt", "improve_prompt.txt",
        "initial_prompt.txt", "config.json"
    ]
    for file in required_files:
        if not os.path.exists(f"{configs_dir}{file}"):
            print(f"Ошибка: Файл {configs_dir}{file} не найден. Создайте его с соответствующим содержимым.")
            exit(1)

    # Read describing exercise from initial_promt.txt
    initial_prompt = read_prompt(f"{configs_dir}initial_prompt.txt")
    
    # Read number of 'critic-improve' cycles
    iterations = config["iterations"]

    # Run brainstorming session and save results
    history, execution_time = brainstorm(initial_prompt, iterations=iterations)
    save_results(
        initial_prompt, history, execution_time,
        prompt_files=["generate_prompt.txt", "critique_prompt.txt", "improve_prompt.txt"]
    )