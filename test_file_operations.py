import os
import sys

def test_file_operations():
    print("=== Testing File Operations ===")
    
    # Test creating a directory
    test_dir = 'test_dir'
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✓ Successfully created directory: {os.path.abspath(test_dir)}")
    except Exception as e:
        print(f"✗ Failed to create directory: {e}")
    
    # Test writing to a file
    test_file = 'test_file.txt'
    try:
        with open(test_file, 'w') as f:
            f.write("This is a test file.")
        print(f"✓ Successfully wrote to file: {os.path.abspath(test_file)}")
    except Exception as e:
        print(f"✗ Failed to write to file: {e}")
    
    # Test writing to a file in a subdirectory
    subdir_file = os.path.join(test_dir, 'test_file.txt')
    try:
        with open(subdir_file, 'w') as f:
            f.write("This is a test file in a subdirectory.")
        print(f"✓ Successfully wrote to file: {os.path.abspath(subdir_file)}")
    except Exception as e:
        print(f"✗ Failed to write to subdirectory file: {e}")
    
    # Test listing directory contents
    try:
        print("\nCurrent directory contents:")
        for item in os.listdir('.'):
            print(f"- {item} (file: {os.path.isfile(item)}, dir: {os.path.isdir(item)})")
    except Exception as e:
        print(f"✗ Failed to list directory contents: {e}")
    
    # Cleanup
    try:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n✓ Cleaned up test file: {test_file}")
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(test_dir)
            print(f"✓ Cleaned up test directory: {test_dir}")
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_file_operations()
