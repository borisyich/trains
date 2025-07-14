import asyncio
import logging
import nest_asyncio
import ollama
import os
import pydub
import telegram
import torch
import warnings
from datetime import datetime
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import pipeline

# Suppress warnings to keep logs clean
warnings.filterwarnings("ignore")

# Apply nest_asyncio to allow nested event loops, necessary for running asyncio in environments like Jupyter
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

# Configure logging with timestamp, level, and message format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping of punctuation labels to their respective symbols
PUNCTUATION_MAP = {
    "COMMA": ", ",
    "PERIOD": ". ",
    "QUESTION": "? ",
    "EXCLAMATION": "! ",
    "NONE": " "
}

def restore_punctuation(text, punctuation_model):
    """
    Restore punctuation in text using a pre-trained punctuation model.
    
    Args:
        text (str): Input text without punctuation.
        punctuation_model: Pre-trained model for punctuation restoration.
    
    Returns:
        str: Text with restored punctuation and proper capitalization.
    """
    try:
        # Tokenize and predict punctuation for each chunk
        predictions = punctuation_model(text, aggregation_strategy="simple")
        result = ""
        capitalize_next = True  # Capitalize first word of the chunk
        for pred in predictions:
            word = pred["word"]
            punctuation = pred["entity"]
            # Capitalize word if needed
            if capitalize_next:
                word = word.capitalize()
                capitalize_next = False
            result += word
            # Add corresponding punctuation
            result += PUNCTUATION_MAP.get(punctuation, " ")
            # Capitalize next word after sentence-ending punctuation
            if punctuation in ["PERIOD", "QUESTION", "EXCLAMATION"]:
                capitalize_next = True
        return result.strip()
    except Exception as e:
        logging.error(f"Error in punctuation restoration: {e}, text: {text}")
        return text  # Return original text on error

# Initialize speech recognition and punctuation restoration models
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0)
punctuation_model = pipeline("token-classification", model="oliverguhr/fullstop-punctuation-multilang-large", device=0)

# Directory for saving transcription files
OUTPUT_DIR = "transcriptions"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

async def start(update, context):
    """
    Handle the /start command, sending a welcome message to the user.
    
    Args:
        update: Telegram update object containing message details.
        context: Telegram context object for bot interactions.
    """
    await update.message.reply_text("Send a voice message, and I will transcribe it, restore punctuation, and generate a summary.")

async def handle_voice(update, context):
    """
    Process incoming voice messages, transcribe them, restore punctuation, summarize, and save results.
    
    Args:
        update: Telegram update object containing the voice message.
        context: Telegram context object for bot interactions.
    """
    voice = update.message.voice

    # Download voice message
    file = await context.bot.get_file(voice.file_id)
    file_path = os.path.join(OUTPUT_DIR, f"voice_{voice.file_id}.ogg")
    await file.download_to_drive(file_path)

    # Convert OGG to WAV for Whisper model compatibility
    wav_path = file_path.replace(".ogg", ".wav")
    audio = pydub.AudioSegment.from_ogg(file_path)
    audio.export(wav_path, format="wav")

    # Transcribe audio to text
    logging.info("Starting transcription...")
    result = whisper(wav_path, return_timestamps=True)
    text = result["text"]
    chunks = result["chunks"]  # Segments with timestamps

    # Restore punctuation in transcribed text
    logging.info("Restoring punctuation...")
    punctuated_text = restore_punctuation(text, punctuation_model)

    # Format transcription with timestamps
    timestamped_text = ""
    for chunk in chunks:
        start_time = chunk["timestamp"][0]
        end_time = chunk["timestamp"][1]
        chunk_text = chunk["text"]
        # Restore punctuation for each chunk
        punctuated_chunk = restore_punctuation(chunk_text, punctuation_model)
        timestamped_text += f"[{start_time:.2f} - {end_time:.2f}] {punctuated_chunk}\n"

    # Summarize text using Gemma3n via Ollama
    logging.info("Summarizing text...")
    summary_prompt = f"Напиши мне саммари по тексту в трёх-пяти предложениях на русском языке:\n{punctuated_text}"
    try:
        response = ollama.generate(model="gemma3n", prompt=summary_prompt, options={"num_predict": 200})
        summary = response["response"]
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        summary = "Error generating summary."

    # Save transcription and summary to Markdown file
    timestamp = datetime.now()
    timestamp_for_name = timestamp.strftime("%Y%m%d_%H%M%S")
    timestamp_for_summary = timestamp.strftime('%B %d, %Y')
    md_file = os.path.join(OUTPUT_DIR, f"transcription_{timestamp_for_name}.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Транскрипция голосового сообщения от {timestamp_for_summary}\n\n")
        f.write("## Полный текст\n")
        f.write(timestamped_text)
        f.write("\n## Краткое содержание\n")
        f.write(summary)

    # Send results to user
    await update.message.reply_text(f"Transcription completed. Full text:\n{punctuated_text}\n\nSummary:\n{summary}")
    with open(md_file, "rb") as f:
        await context.bot.send_document(chat_id=update.message.chat_id, document=f)

    # Clean up temporary files
    os.remove(file_path)
    os.remove(wav_path)

async def main():
    """
    Main function to set up and run the Telegram bot.
    """
    # Initialize Telegram bot with token
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    # Register command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    # Start polling for updates
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())