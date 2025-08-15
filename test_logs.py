import os
import sys

def test_logs():
    print("=== Testing Log File Creation ===")
    
    # Try to create logs directory
    logs_dir = 'logs'
    try:
        os.makedirs(logs_dir, exist_ok=True)
        print(f"✓ Created logs directory: {os.path.abspath(logs_dir)}")
    except Exception as e:
        print(f"✗ Failed to create logs directory: {e}")
        return
    
    # Try to write to a log file
    log_file = os.path.join(logs_dir, 'test.log')
    try:
        with open(log_file, 'w') as f:
            f.write("This is a test log entry.\n")
        print(f"✓ Successfully wrote to log file: {os.path.abspath(log_file)}")
        
        # Verify the file was created and has content
        if os.path.exists(log_file):
            print(f"✓ Log file exists at: {os.path.abspath(log_file)}")
            with open(log_file, 'r') as f:
                content = f.read()
            print(f"✓ Log file content: {content.strip()}")
        else:
            print("✗ Log file was not created")
    except Exception as e:
        print(f"✗ Failed to write to log file: {e}")
    
    print("\nCurrent directory contents:")
    try:
        for item in os.listdir('.'):
            print(f"- {item} (file: {os.path.isfile(item)}, dir: {os.path.isdir(item)})")
            
        if os.path.exists(logs_dir):
            print(f"\nContents of {logs_dir}:")
            for item in os.listdir(logs_dir):
                print(f"- {item}")
    except Exception as e:
        print(f"Error listing directory contents: {e}")

if __name__ == "__main__":
    test_logs()
