#!/usr/bin/env python3
"""
Verify Streamlit configuration is properly loaded
"""
import streamlit as st

def verify_config():
    """Verify config values are properly set"""
    print("🔧 Streamlit Configuration Verification")
    print("=" * 50)
    
    # Get config values
    try:
        from streamlit import config
        max_upload = config.get_option("server.maxUploadSize")
        max_message = config.get_option("server.maxMessageSize") 
        enable_cors = config.get_option("server.enableCORS")
        enable_xsrf = config.get_option("server.enableXsrfProtection")
        
        print(f"📤 Max Upload Size: {max_upload} MB")
        print(f"💬 Max Message Size: {max_message} MB") 
        print(f"🌐 Enable CORS: {enable_cors}")
        print(f"🛡️ Enable XSRF Protection: {enable_xsrf}")
        
        # Verify our settings
        if max_upload >= 3072 and max_message >= 3072:
            print("✅ Configuration loaded successfully - 3GB limits active!")
            return True
        else:
            print("❌ Configuration not loaded properly - still using defaults")
            return False
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

if __name__ == "__main__":
    verify_config()