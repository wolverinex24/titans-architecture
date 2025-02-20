import os

def generate_directory_structure(start_path, output_file):
    """
    Generate a text file containing the directory structure starting from the given path.
    
    Args:
        start_path (str): Root directory path to start scanning from
        output_file (str): Name of the output text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Directory structure for: {os.path.abspath(start_path)}\n")
        f.write("=" * 50 + "\n\n")
        
        for root, dirs, files in os.walk(start_path):
            # Calculate the current depth to determine indentation
            level = root.replace(start_path, '').count(os.sep)
            indent = '│   ' * level
            
            # Write the current directory
            folder_name = os.path.basename(root)
            if level > 0:  # Don't write the root folder as a subfolder
                f.write(f"{indent}├── {folder_name}/\n")
            
            # Write all files in the current directory
            file_indent = '│   ' * (level + 1)
            for i, file in enumerate(sorted(files)):
                if i == len(files) - 1 and len(dirs) == 0:  # Last file in last directory
                    f.write(f"{file_indent}└── {file}\n")
                else:
                    f.write(f"{file_indent}├── {file}\n")

if __name__ == "__main__":
    # Use the current directory as the starting point
    current_dir = "."
    output_filename = "directory_structure.txt"
    
    try:
        generate_directory_structure(current_dir, output_filename)
        print(f"Directory structure has been saved to {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")