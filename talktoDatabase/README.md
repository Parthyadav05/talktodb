# TalkDB - Natural Language Database Interface

## Installation

1. Run `install.bat` to install Python dependencies
2. Edit `.env` file with your database configuration
3. Run `TalkDB.bat` to start the server
4. Access the application at http://localhost:8080

## Configuration

Edit the `.env` file with your settings:

- DB_HOST: Your database host
- DB_USER: Your database username  
- DB_PASSWORD: Your database password
- DB_NAME: Your database name
- GOOGLE_API_KEY: Your Google Gemini API key

## Usage

1. Start: Double-click `TalkDB.bat`
2. Open browser: http://localhost:8080
3. API docs: http://localhost:8080/docs
4. Health check: http://localhost:8080/health

## Requirements

- Python 3.8+
- PostgreSQL database
- Google Gemini API key
- Internet connection
