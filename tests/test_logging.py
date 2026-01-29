import unittest
import os
import shutil
from pathlib import Path
import sys
import logging
import time

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logging_config import setup_logging

class TestLoggingSystem(unittest.TestCase):
    def setUp(self):
        # Clean up logs directory before each test
        self.log_dir = Path("logs")
        if self.log_dir.exists():
            # Close all handlers to avoid permission errors on Windows
            for logger_name in logging.root.manager.loggerDict:
                logger = logging.getLogger(logger_name)
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            
            # Wait a bit for file handles to be released
            time.sleep(0.1)
            try:
                shutil.rmtree(self.log_dir)
            except Exception as e:
                print(f"Warning: could not clean up logs: {e}")
        
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def test_categorical_logging(self):
        # Setup specific loggers
        logger_agents = setup_logging("agents")
        logger_rag = setup_logging("rag")
        
        # Write some logs
        logger_agents.info("Testing agents info message")
        logger_rag.info("Testing rag info message")
        
        # Verify files exist
        self.assertTrue((self.log_dir / "agents.log").exists())
        self.assertTrue((self.log_dir / "rag.log").exists())
        
        # Check contents
        with open(self.log_dir / "agents.log", "r") as f:
            content = f.read()
            self.assertIn("Testing agents info message", content)
            self.assertNotIn("Testing rag info message", content)
            
        with open(self.log_dir / "rag.log", "r") as f:
            content = f.read()
            self.assertIn("Testing rag info message", content)
            self.assertNotIn("Testing agents info message", content)

    def test_error_aggregation(self):
        logger_api = setup_logging("api")
        
        # Info should NOT go to errors.log
        logger_api.info("This is an info message")
        
        # Error SHOULD go to both api.log and errors.log
        logger_api.error("This is an error message")
        
        self.assertTrue((self.log_dir / "api.log").exists())
        self.assertTrue((self.log_dir / "errors.log").exists())
        
        with open(self.log_dir / "errors.log", "r") as f:
            content = f.read()
            self.assertIn("This is an error message", content)
            self.assertNotIn("This is an info message", content)

    def test_contextual_info(self):
        logger = setup_logging("app")
        logger.info("Contextual test")
        
        with open(self.log_dir / "app.log", "r") as f:
            content = f.read()
            # Expecting format: time | LEVEL | category:function:line - message
            # name in logger should be "app"
            self.assertIn("| INFO     | app:test_contextual_info:", content)

if __name__ == "__main__":
    unittest.main()
