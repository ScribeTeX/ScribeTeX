"""
ScribeTeX Configuration
Environment-based configuration for the application.
"""
import os
from dotenv import load_dotenv
import secrets

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # LLM Provider settings
    # Supported providers: 'openai', 'google', 'anthropic'
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai').lower()
    
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Model configuration
    # OpenAI models (Jan 2026):
    #   - Non-reasoning: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
    #   - Reasoning: gpt-5, gpt-5.1, gpt-5.2 (use reasoning_effort parameter)
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4.1')
    
    # Google models (Jan 2026):
    #   - gemini-2.5-flash, gemini-2.5-pro (stable)
    #   - gemini-3-flash-preview, gemini-3-pro-preview (preview)
    GOOGLE_MODEL = os.environ.get('GOOGLE_MODEL', 'gemini-2.5-flash')
    
    # Anthropic models (Jan 2026):
    #   - claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5
    #   - Extended thinking available via thinking parameter
    ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-5')

    # PDF compilation settings
    COMPILE_PDF_ENABLED = os.environ.get('COMPILE_PDF_ENABLED', 'True').lower() == 'true'

    # File storage directories
    UPLOAD_BASE_FOLDER = 'uploads'
    OUTPUT_BASE_FOLDER = 'output'

    # Session settings
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'flask_session')
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True

    # Maximum upload size (100 MB)
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    
    # Supported file extensions
    ALLOWED_EXTENSIONS = {
        'pdf': ['.pdf'],
        'image': ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'],
        'text': ['.txt', '.md', '.markdown', '.tex', '.rst'],
        'document': ['.docx', '.doc', '.rtf', '.odt']
    }
    
    @classmethod
    def get_all_allowed_extensions(cls):
        """Return a flat set of all allowed file extensions."""
        extensions = set()
        for ext_list in cls.ALLOWED_EXTENSIONS.values():
            extensions.update(ext_list)
        return extensions
