import asyncio
import pandas as pd
import json
import logging
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, JobQueue, MessageHandler, filters, ContextTypes

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration constants
BOT_TOKEN = os.getenv('BOT_TOKEN')  # Telegram bot token from environment
BASE_URL = os.getenv('BASE_URL')  # Base URL for job vacancy scraping
JSON_FILE = os.getenv('JSON_FILE')  # Path to store scraped vacancies in JSON format
CSV_FILE = os.getenv('CSV_FILE')  # Path to store vacancy statistics in CSV format

def load_vacancies():
    """
    Loads previously saved job vacancies from a JSON file.
    
    Returns:
        dict: A dictionary containing vacancy data if the file exists and is valid, otherwise an empty dictionary.
    """
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {JSON_FILE}: {e}")
        return {}

def save_vacancies(vacancies):
    """
    Saves job vacancies to a JSON file.
    
    Args:
        vacancies (dict): A dictionary of vacancies to save, with URLs as keys and vacancy details as values.
    """
    try:
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(vacancies, f, ensure_ascii=False, indent=4)
        logger.info(f"Vacancies saved to {JSON_FILE}")
    except Exception as e:
        logger.error(f"Error saving {JSON_FILE}: {e}")

def append_to_csv(vacancies):
    """
    Appends vacancy data to a CSV file with a timestamp.
    
    Args:
        vacancies (dict): A dictionary of vacancies to append, with URLs as keys and vacancy details as values.
    """
    try:
        # Prepare data for CSV
        data = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for url, vacancy in vacancies.items():
            data.append({
                'url': url,
                'title': vacancy['title'],
                'salary': vacancy['salary'],
                'company': vacancy['company'],
                'timestamp': current_time
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Append to CSV (append mode, no header overwrite)
        if os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(CSV_FILE, mode='w', header=True, index=False, encoding='utf-8')
        
        logger.info(f"Appended {len(data)} records to {CSV_FILE}")
    except Exception as e:
        logger.error(f"Error appending to {CSV_FILE}: {e}")

async def parse_vacancies(context: ContextTypes.DEFAULT_TYPE, chat_id: int = None):
    """
    Parses job vacancies from all pages of the target website and sends progress updates to the chat.
    
    Args:
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
        chat_id (int, optional): The Telegram chat ID to send progress updates to. Defaults to None.
    
    Returns:
        dict: A dictionary of parsed vacancies with URLs as keys and vacancy details as values.
    """
    vacancies = {}
    page = 1
    while True:
        url = f"{BASE_URL}&page={page}" if page > 1 else BASE_URL
        try:
            logger.info(f"Parsing page {page}: {url}")
            if chat_id is not None:
                await context.bot.send_message(chat_id=chat_id, text=f"Парсинг страницы {page}...")
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to load page {url}: status {response.status_code}")
                if chat_id is not None:
                    await context.bot.send_message(
                        chat_id=chat_id, text=f"Ошибка: страница {page} недоступна ({response.status_code}).")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            vacancy_cards = soup.find_all('div', class_='fui-flex fui-flex-col fui-bg-white fui-sheet-shadow_sm fui-card transition-shadow cursor-pointer')

            if not vacancy_cards:
                logger.info(f"No vacancies found on page {page}")
                if chat_id is not None:
                    await context.bot.send_message(
                        chat_id=chat_id, text=f"Не найдено вакансий на странице {page}. Парсинг завершен.")
                break

            for card in vacancy_cards:
                title_elem = card.find('a', class_='fui-no-underline font-bold text-lg w-0 grow flex-wrap text-blue-dark')
                title = title_elem.text.strip() if title_elem else "N/A"
                vacancy_url = title_elem['href'] if title_elem else ""
                if vacancy_url and not vacancy_url.startswith('http'):
                    vacancy_url = "https://finder.work" + vacancy_url
                
                salary_elem = card.find('div', class_='font-bold text-black text-xl')
                salary = salary_elem.text.replace('\xa0', ' ') if salary_elem else "N/A"
                
                company_elem = card.find('a', class_='fui-no-underline text-grey-dark')
                company = company_elem.text if company_elem else "N/A"

                if vacancy_url and title != "N/A":  # Валидация данных
                    vacancies[vacancy_url] = {
                        'title': title.strip(),
                        'salary': salary.strip() if salary != "N/A" else "Не указана",
                        'company': company.strip() if company != "N/A" else "Не указана",
                        'url': vacancy_url
                    }

            page += 1
        except requests.RequestException as e:
            logger.error(f"Error requesting page {url}: {e}")
            if chat_id is not None:
                await context.bot.send_message(chat_id=chat_id, text=f"Ошибка при загрузке страницы {page}: {str(e)}")
            break

    if vacancies:
        append_to_csv(vacancies)
        save_vacancies(vacancies)
        logger.info(f"Saved {len(vacancies)} vacancies to {JSON_FILE} and {CSV_FILE}")
        if chat_id is not None:
            await context.bot.send_message(chat_id=chat_id, text=f"Сохранено {len(vacancies)} вакансий.")
    return vacancies

async def send_welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sends a welcome message to the user if they interact with the bot for the first time.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages and storing chat data.
    """
    chat_id = update.message.chat_id
    if not context.chat_data.get('welcome_sent', False):
        welcome_message = (
            "Добро пожаловать! Я - бот для отслеживания вакансий для подработки. Напишите /list, чтобы увидеть доступные команды."
        )
        await context.bot.send_message(chat_id=chat_id, text=welcome_message)
        context.chat_data['welcome_sent'] = True
        logger.info(f"Sent welcome message to chat {chat_id}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /start command: parses vacancies, saves them, and sends the JSON file to the user.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received /start command from chat {chat_id}")

    # Send welcome message if not already sent
    await send_welcome_message(update, context)

    await context.bot.send_message(chat_id=chat_id, text="Начинаю парсинг вакансий...")

    # Parse vacancies
    vacancies = await parse_vacancies(context, chat_id)

    if not vacancies:
        await context.bot.send_message(chat_id=chat_id, 
                                       text="Не удалось найти вакансии. Возможно, проблема с сайтом.")
        logger.warning("Parsing completed: no vacancies found")
        return

    # Append to CSV before saving JSON
    append_to_csv(vacancies)

    # Save vacancies
    save_vacancies(vacancies)
    await context.bot.send_message(chat_id=chat_id, 
                                   text=f"Найдено {len(vacancies)} вакансий. Сохранено в {JSON_FILE} \
                                    и добавлено в {CSV_FILE}. Отправляю файл {JSON_FILE}...")

    # Send JSON file
    try:
        with open(JSON_FILE, 'rb') as f:
            await context.bot.send_document(chat_id=chat_id, document=f, filename=JSON_FILE)
        logger.info(f"File {JSON_FILE} sent to chat {chat_id}")
    except Exception as e:
        logger.error(f"Error sending file: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Ошибка при отправке файла.")

async def check_vacancies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /check command: compares current vacancies with saved ones and sends new ones.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received /check command from chat {chat_id}")

    # Send welcome message if not already sent
    await send_welcome_message(update, context)

    await context.bot.send_message(chat_id=chat_id, text="Проверяю новые вакансии...")

    # Load existing vacancies
    old_vacancies = load_vacancies()
    if not old_vacancies:
        await context.bot.send_message(
            chat_id=chat_id, text="Предыдущие вакансии не найдены. Используйте /start или загрузите JSON-файл.")
        return

    # Parse current vacancies
    new_vacancies = await parse_vacancies(context, chat_id)

    if not new_vacancies:
        await context.bot.send_message(
            chat_id=chat_id, text="Не удалось найти вакансии. Возможно, проблема с сайтом.")
        return

    # Check for new vacancies
    new_vacancy_found = False
    for url, vacancy in new_vacancies.items():
        if url not in old_vacancies:
            message = f"Новая вакансия!\n\nНазвание: {vacancy['title']}\n\
                Зарплата: {vacancy['salary']}\nОрганизация: {vacancy['company']}\nСсылка: {url}"
            await context.bot.send_message(chat_id=chat_id, text=message)
            new_vacancy_found = True
            logger.info(f"New vacancy found: {vacancy['title']}")

    if not new_vacancy_found:
        await context.bot.send_message(chat_id=chat_id, text="Новых вакансий не найдено.")
        logger.info("No new vacancies found")

    # Append to CSV before saving JSON
    append_to_csv(new_vacancies)

    # Save new data
    save_vacancies(new_vacancies)
    await context.bot.send_message(
        chat_id=chat_id, text=f"Обновлён файл {JSON_FILE} с {len(new_vacancies)} вакансиями. \
            Добавлено в {CSV_FILE}.")

async def get_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /getfile command: sends the current vacancies.json file.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received /getfile command from chat {chat_id}")

    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'rb') as f:
                await context.bot.send_document(chat_id=chat_id, document=f, filename=JSON_FILE)
            await context.bot.send_message(chat_id=chat_id, text=f"Файл {JSON_FILE} отправлен.")
            logger.info(f"File {JSON_FILE} sent to chat {chat_id}")
        except Exception as e:
            logger.error(f"Error sending file: {e}")
            await context.bot.send_message(chat_id=chat_id, text="Ошибка при отправке файла.")
    else:
        await context.bot.send_message(
            chat_id=chat_id, text="Файл vacancies.json не найден. Используйте /start для создания нового файла.")
        logger.warning("File vacancies.json not found during /getfile")

async def get_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /getcsv command: sends the current vacancies_stats.csv file.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received /getcsv command from chat {chat_id}")

    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, 'rb') as f:
                await context.bot.send_document(chat_id=chat_id, document=f, filename=CSV_FILE)
            await context.bot.send_message(chat_id=chat_id, text=f"Файл {CSV_FILE} отправлен.")
            logger.info(f"File {CSV_FILE} sent to chat {chat_id}")
        except Exception as e:
            logger.error(f"Error sending file: {e}")
            await context.bot.send_message(chat_id=chat_id, text="Ошибка при отправке файла.")
    else:
        await context.bot.send_message(
            chat_id=chat_id, text=f"Файл {CSV_FILE} не найден. Используйте /start или /check для создания.")
        logger.warning(f"File {CSV_FILE} not found during /getcsv")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /status command: checks if the bot is running and shows the current time.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received /status command from chat {chat_id}")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await context.bot.send_message(chat_id=chat_id, text=f"Бот работает! Текущее время: {current_time}")

async def list_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /list command: displays a list of available bot commands.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received /list command from chat {chat_id}")

    message = (
        "Доступные команды бота:\n\n"
        "/start - Парсит текущие вакансии, сохраняет их в vacancies.json и добавляет в vacancies_stats.csv, затем отправляет JSON-файл.\n"
        "/check - Проверяет новые вакансии, сравнивая с vacancies.json, отправляет новые, обновляет JSON и добавляет в CSV.\n"
        "/getfile - Отправляет текущий файл vacancies.json.\n"
        "/getcsv - Отправляет текущий файл vacancies_stats.csv.\n"
        "/status - Проверяет, работает ли бот, и показывает текущее время.\n"
        "Отправка JSON-файла - Загрузите JSON-файл с вакансиями (например, вчерашний vacancies.json), чтобы использовать его для сравнения при /check."
    )
    await context.bot.send_message(chat_id=chat_id, text=message)
    logger.info("Command list sent to chat")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles uploaded JSON files containing vacancy data for comparison.
    
    Args:
        update (Update): The Telegram update object containing the uploaded file.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    file = update.message.document
    logger.info(f"Received file from chat {chat_id}: {file.file_name}")

    if file.mime_type != 'application/json':
        await context.bot.send_message(chat_id=chat_id, text="Пожалуйста, загрузите файл в формате JSON.")
        logger.warning(f"Received file with incorrect MIME type: {file.mime_type}")
        return

    await context.bot.send_message(chat_id=chat_id, text="Получен JSON-файл, обрабатываю...")

    try:
        # Download the file
        file_obj = await file.get_file()
        file_path = await file_obj.download_to_drive(JSON_FILE)

        # Validate JSON file
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            json.load(f)  # Check if the file is valid JSON

        await context.bot.send_message(
            chat_id=chat_id, text=f"Файл {JSON_FILE} успешно загружен \
                и будет использован для сравнения при следующей команде /check.")
        logger.info(f"File {JSON_FILE} successfully uploaded from chat {chat_id}")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        await context.bot.send_message(
            chat_id=chat_id, text=f"Ошибка при загрузке файла: {str(e)}. Убедитесь, что это валидный JSON-файл.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles non-command text messages by prompting the user to use commands.
    
    Args:
        update (Update): The Telegram update object containing message details.
        context (ContextTypes.DEFAULT_TYPE): Telegram bot context for sending messages.
    """
    chat_id = update.message.chat_id
    logger.info(f"Received text message from chat {chat_id}")

    await context.bot.send_message(chat_id=chat_id, text="Пожалуйста, используйте команды. Напишите /list, чтобы увидеть доступные команды.")

def main():
    """
    Initializes and runs the Telegram bot, checking for the existence of the vacancies JSON file on startup.
    """
    # Create a JobQueue instance
    job_queue = JobQueue()

    # Build the application with the JobQueue
    application = Application.builder().token(BOT_TOKEN).job_queue(job_queue).build()

    # Register command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("check", check_vacancies))
    application.add_handler(CommandHandler("getfile", get_file))
    application.add_handler(CommandHandler("getcsv", get_csv))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("list", list_commands))
    application.add_handler(MessageHandler(filters.Document.MimeType("application/json"), handle_file))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Ensure data directory exists
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)

    # Check if vacancies.json exists; if not, schedule initial parsing
    if not os.path.exists(JSON_FILE):
        logger.info("File vacancies.json not found, scheduling initial parsing")
        application.job_queue.run_once(parse_vacancies, when=0, data={'chat_id': None})

    # Start the bot
    logger.info("Bot successfully started")
    application.run_polling()

if __name__ == "__main__":
    main()