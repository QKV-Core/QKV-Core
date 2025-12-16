"""
Application state management.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
# This must be done before any imports from parent directories
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
_web_ui_root = _current_file.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_web_ui_root) not in sys.path:
    sys.path.insert(0, str(_web_ui_root))

from qkv_core.storage.postgresql_db import PostgreSQLManager


class AppState:
    """Manages application-wide state including database connection and model references."""
    
    def __init__(self):
        # Import logger here to ensure path is set
        try:
            from qkv_core.utils.logger import get_logger
        except ImportError as e:
            import sys
            from pathlib import Path
            print(f"❌ Import error: {e}")
            print(f"Current sys.path: {sys.path[:5]}")
            print(f"Project root should be: {Path(__file__).parent.parent.parent}")
            print(f"utils/logger.py should be at: {Path(__file__).parent.parent.parent / 'utils' / 'logger.py'}")
            raise
        logger = get_logger()
        
        postgresql_connection_string = os.getenv('POSTGRESQL_CONNECTION_STRING')
        if not postgresql_connection_string:
            try:
                from config.database_config import get_postgresql_connection_string
                postgresql_connection_string = get_postgresql_connection_string()
            except ImportError:
                # Fallback if config module not available
                postgresql_connection_string = "host=localhost dbname=llm_core user=postgres password=postgres"
            logger.warning("⚠️  POSTGRESQL_CONNECTION_STRING not set, using default: host=localhost dbname=llm_core user=postgres")
        
        try:
            self.db = PostgreSQLManager(postgresql_connection_string)
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.error("💡 Please ensure PostgreSQL is running and POSTGRESQL_CONNECTION_STRING is set correctly")
            raise RuntimeError(f"PostgreSQL connection failed: {e}. Please check your .env file and PostgreSQL service.")
        
        self.current_tokenizer = None
        self.current_model = None
        self.current_trainer = None
        self.inference_engine = None
        self.training_active = False
        self.auto_training_complete = False
        self.auto_trained_checkpoint = None
        self.auto_trained_tokenizer = None
        self.model_quantized = False


# Global state instance
state = AppState()

# Global model variables (legacy support)
model = None
tokenizer = None
device = None
hybrid_engine = None
hybrid_model_ids = []
hybrid_mode = 'ensemble'
model_config = None
