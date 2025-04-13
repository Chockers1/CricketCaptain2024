import streamlit as st

def setup_secrets():
    """
    Run this once to set up credentials in Streamlit's secrets.
    This is typically used for deployment environments.
    """
    print("Setting up credentials in Streamlit's secrets.")
    
    # Create the .streamlit directory if it doesn't exist
    import os
    import dotenv
    
    # Load from .env if available
    dotenv.load_dotenv()
    
    # Default values that can be overridden
    default_username = os.environ.get('CC_USERNAME', 'CC')
    default_password = os.environ.get('CC_PASSWORD', 'CCapril2025')
    
    # Get values from user input
    username = input(f"Enter username (default: {default_username}): ") or default_username
    password = input(f"Enter password (default: {default_password}): ") or default_password
    
    # Create the .streamlit directory
    os.makedirs('.streamlit', exist_ok=True)
    
    # Create or update the secrets.toml file
    with open('.streamlit/secrets.toml', 'w') as f:
        f.write(f"""[login]
username = "{username}"
password = "{password}"
""")
    
    print("Credentials have been saved to .streamlit/secrets.toml")
    print("This file should NOT be committed to GitHub.")

if __name__ == "__main__":
    setup_secrets()
