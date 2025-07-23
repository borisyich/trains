# Part_time_job_parser

## Overview
This is a Telegram bot designed to scrape job vacancies from [finder.work](https://finder.work), save them in JSON and CSV formats, and notify users about new vacancies. The bot supports commands to parse vacancies, check for new listings, and retrieve saved data files.

## Features
- **Scrape Vacancies**: Fetches job listings from finder.work, including title, salary, company, and URL.
- **Save Data**: Stores vacancies in a JSON file (`vacancies.json`) and appends to a CSV file (`vacancies_stats.csv`) with timestamps.
- **Check New Vacancies**: Compares current vacancies with previously saved ones and notifies users of new listings.
- **File Retrieval**: Allows users to download the latest JSON and CSV files.
- **File Upload**: Users can upload a JSON file to use for comparison with new vacancies.
- **Status Check**: Confirms the bot is running and displays the current time.
- **Command Listing**: Provides a list of available commands via `/list`.

## Requirements
- Python 3.8+
- Libraries:
  - `python-telegram-bot`
  - `requests`
  - `beautifulsoup4`
  - `pandas`
  - `python-dotenv`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/borisyich/trains.git
   cd trains/part_time_job_parser
   ```

2. **Create a `.env` File**:
   Create a `.env` file in the project root with the following content:
   ```env
   BOT_TOKEN=your_telegram_bot_token
   ```
   Obtain a bot token from [BotFather](https://t.me/BotFather) on Telegram.

3. **Create Data Directory**:
   Create a `data` directory in the project root to store `vacancies.json` and `vacancies_stats.csv`:
   ```bash
   mkdir data
   ```

## Usage
1. **Run the Bot**:
   ```bash
   python app.py
   ```

2. **Interact with the Bot**:
   - Start a chat with your bot on Telegram.
   - Use the following commands:
     - `/start`: Parse current vacancies, save them, and send the JSON file.
     - `/check`: Compare current vacancies with saved ones and notify about new ones.
     - `/getfile`: Retrieve the latest `vacancies.json` file.
     - `/getcsv`: Retrieve the latest `vacancies_stats.csv` file.
     - `/status`: Check if the bot is running and see the current time.
     - `/list`: List all available commands.
     - Upload a JSON file to use for comparison with `/check`.

## Environment Setup
- Ensure the `.env` file contains a valid `BOT_TOKEN`.
- The bot requires internet access to scrape vacancies from finder.work.
- The `data` directory must exist for saving JSON and CSV files.
- The bot logs events to the console for debugging (INFO level).

## Notes
- The bot scrapes vacancies from `https://finder.work/vacancies/project?categories=1`. Ensure the website structure hasn't changed, as this may break the parsing logic.
- The bot uses asynchronous programming (`asyncio`) for Telegram interactions.
- Error handling is implemented for network issues, file operations, and JSON validation.
- The bot assumes UTF-8 encoding for file operations.
