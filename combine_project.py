import os

def combine_files_to_single_file(root_dir, output_filename="combined_project_files.txt"):
    """
    Combines the content of all files in a directory into a single file,
    excluding 'node_modules' directories and 'package-lock.json' files.

    Args:
        root_dir (str): The root directory of the project.
        output_filename (str): The name of the output file.
    """
    combined_content = []
    excluded_dirs = ['node_modules', '.git', '.vscode', '.idea', 'dist', 'build']
    excluded_files = ['package-lock.json', 'yarn.lock', 'bun.lockb', '.env'] # Added .env as it might contain sensitive info

    print(f"Starting to combine files from: {root_dir}")
    print(f"Excluding directories: {excluded_dirs}")
    print(f"Excluding files: {excluded_files}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

        for filename in filenames:
            if filename in excluded_files:
                print(f"Skipping excluded file: {os.path.join(dirpath, filename)}")
                continue

            file_path = os.path.join(dirpath, filename)
            # Make path relative to root_dir for the header
            relative_file_path = os.path.relpath(file_path, root_dir)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                combined_content.append(f"\n--- START FILE: {relative_file_path} ---\n\n")
                combined_content.append(content)
                combined_content.append(f"\n\n--- END FILE: {relative_file_path} ---\n")
                print(f"Included file: {relative_file_path}")
            except UnicodeDecodeError:
                print(f"Skipping binary or undecodable file (UnicodeDecodeError): {relative_file_path}")
            except Exception as e:
                print(f"Error reading file {relative_file_path}: {e}")

    output_path = os.path.join(root_dir, output_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write("".join(combined_content))
        print(f"\nSuccessfully combined all files into: {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")

if __name__ == "__main__":
    # Get the current working directory where the script is run
    # This assumes you run the script from your project's root directory
    project_root = os.getcwd()
    combine_files_to_single_file(project_root)
