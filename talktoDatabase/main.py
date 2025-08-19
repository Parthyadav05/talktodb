#!/usr/bin/env python3
"""
TalkDB - Executable Entry Point
Reads environment variables and starts the server
"""

import os
import sys
import uvicorn
from pathlib import Path

def load_env_file(env_path=".env"):
    """Load environment variables from .env file"""
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"Warning: {env_path} file not found. Using system environment variables.")
        return
    
    print(f"Loading environment variables from {env_path}")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Only set if not already set in system environment
                if not os.getenv(key):
                    os.environ[key] = value

def validate_required_env():
    """Validate required environment variables"""
    required_vars = [
        'DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_NAME',
        'GOOGLE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or system environment.")
        return False
    
    return True

def main():
    """Main entry point for TalkDB executable"""
    print("Starting TalkDB Server...")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_env_file()
    
    # Validate required environment variables
    if not validate_required_env():
        sys.exit(1)
    
    # Set defaults for optional variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    env_mode = os.getenv("ENV", "production")
    
    # Build database URI if not provided
    if not os.getenv("DB_URI"):
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")
        
        db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        os.environ["DB_URI"] = db_uri
    
    # Display configuration
    print("Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Database: {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
    print(f"   Schemas: {os.getenv('ALLOWED_SCHEMAS', 'public')}")
    print(f"   AI Model: {os.getenv('LLM_MODEL', 'gemini-2.5-flash')}")
    print(f"   Environment: {env_mode}")
    print(f"   Log Level: {log_level}")
    print("=" * 50)
    
    # Import and start the FastAPI app
    try:
        print("Initializing application...")
        
        # Start the server
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=(env_mode == "development"),
            access_log=(log_level == "debug")
        )
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"ERROR: Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()