# Telegram Voice Transcription Bot

This project is a Telegram bot that transcribes voice messages, restores punctuation, generates summaries, and saves results in Markdown format.

## Features
- **Voice Transcription**: Converts voice messages to text using the Whisper model (`openai/whisper-large-v3`).
- **Punctuation Restoration**: Adds punctuation to transcribed text using a multilingual punctuation model (`oliverguhr/fullstop-punctuation-multilang-large`).
- **Text Summarization**: Generates a 3-5 sentence summary in Russian using the Gemma3n model via Ollama.
- **Markdown Output**: Saves transcriptions with timestamps and summaries in a Markdown file, which is sent to the user.
- **Temporary File Management**: Automatically converts OGG audio to WAV and cleans up temporary files.

## Requirements
- Python 3.8+
- Libraries:
  - `asyncio`
  - `logging`
  - `nest_asyncio`
  - `ollama`
  - `pydub`
  - `python-telegram-bot`
  - `torch`
  - `transformers`
  - `python-dotenv`
- FFmpeg (for `pydub` audio conversion)
- A `.env` file with `TELEGRAM_BOT_TOKEN` set to your Telegram bot token
- Access to the Gemma3n model via Ollama
- GPU recommended for faster model inference

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/borisyich/trains.git
   cd trains/speech-text-summary
   ```
2. Install dependencies.
3. Install FFmpeg:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
4. Create a `.env` file in the project root with:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```
5. Ensure Ollama is running with the Gemma3n model:
   ```bash
   ollama run gemma3n
   ```

## Usage
1. Run the bot:
   ```bash
   python telegram_bot.py
   ```
2. In Telegram, start the bot with `/start`.
3. Send a voice message to receive:
   - Transcribed text with restored punctuation
   - A 3-5 sentence summary in Russian
   - A Markdown file containing the transcription with timestamps and summary

## Project Structure
- `telegram_bot.py`: Main script containing the bot logic.
- `transcriptions/`: Directory where transcription Markdown files and temporary audio files are stored.
- `.env`: Environment file for storing the Telegram bot token.

## How It Works
1. **Voice Message Handling**:
   - The bot listens for voice messages using the `python-telegram-bot` library.
   - Voice messages are downloaded as OGG files and converted to WAV using `pydub`.
2. **Transcription**:
   - The Whisper model transcribes the audio to text, including timestamps for each segment.
3. **Punctuation Restoration**:
   - The text is split into chunks to avoid model input limits.
   - The punctuation model adds appropriate punctuation and capitalizes words as needed.
4. **Summarization**:
   - The punctuated text is sent to the Gemma3n model via Ollama for summarization.
5. **Output**:
   - Results are formatted into a Markdown file with timestamps and summary.
   - The bot sends the transcribed text, summary, and Markdown file to the user.
6. **Cleanup**:
   - Temporary audio files are deleted after processing.

## Notes
- Ensure your system has sufficient memory and GPU resources for running the Whisper and punctuation models.
- The bot requires an active internet connection for Telegram API and Ollama communication.
- Temporary files are stored in the `transcriptions/` directory and automatically deleted after processing.
- Logs are generated for debugging and monitoring, using the `logging` module.

## Troubleshooting
- **Missing TELEGRAM_BOT_TOKEN**: Ensure the `.env` file contains a valid token from BotFather.
- **FFmpeg errors**: Verify FFmpeg is installed and accessible in your system's PATH.
- **Ollama errors**: Ensure the Gemma3n model is available and Ollama is running.
- **Model errors**: Check that your system has a compatible GPU and sufficient memory for the Whisper and punctuation models.
