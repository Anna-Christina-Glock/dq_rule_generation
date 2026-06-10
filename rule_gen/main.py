"""
Main Entry Point for Rule Generation

This script serves as the main entry point for running the rule generation pipeline.
"""

import logging
from pathlib import Path

from .main_server_parallel import main_rule_gen

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting Rule Generation Pipeline")
    
    # Default configuration paths
    config_path = Path("configs/parameter.json")
    connection_config_path = Path("configs/connection.json")
    
    # Run the main rule generation function
    main_rule_gen(
        config_path=config_path,
        connection_config_path=connection_config_path
    )
    
    logger.info("Rule Generation Pipeline Completed")