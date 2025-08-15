import os
import sys
import logging

def main():
    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_python.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*50)
    logger.info("PYTHON ENVIRONMENT TEST")
    logger.info("="*50)
    
    # Log environment information
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Test file writing
    test_file = "test_output.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Test successful!")
        logger.info(f"Successfully wrote to {test_file}")
        
        # Read the file back
        with open(test_file, 'r') as f:
            content = f.read()
        logger.info(f"File content: {content}")
        
        # Delete the test file
        os.remove(test_file)
        logger.info(f"Successfully deleted {test_file}")
    except Exception as e:
        logger.error(f"File system test failed: {e}")
    
    # List directory contents
    logger.info("\nCurrent directory contents:")
    for item in os.listdir('.'):
        item_path = os.path.join('.', item)
        logger.info(f"- {item} (file: {os.path.isfile(item_path)}, dir: {os.path.isdir(item_path)})")
    
    # Check output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"\nOutput directory: {os.path.abspath(output_dir)}")
    logger.info(f"Output directory exists: {os.path.exists(output_dir)}")
    logger.info(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")
    logger.info(f"Contents of output directory: {os.listdir(output_dir) if os.path.exists(output_dir) else 'N/A'}")
    
    # Test writing to output directory
    test_output_file = os.path.join(output_dir, 'test_output.txt')
    try:
        with open(test_output_file, 'w') as f:
            f.write("Test successful!")
        logger.info(f"Successfully wrote to {test_output_file}")
        
        # Read the file back
        with open(test_output_file, 'r') as f:
            content = f.read()
        logger.info(f"Output file content: {content}")
        
        # Delete the test file
        os.remove(test_output_file)
        logger.info(f"Successfully deleted {test_output_file}")
    except Exception as e:
        logger.error(f"Output directory test failed: {e}")

if __name__ == "__main__":
    main()
