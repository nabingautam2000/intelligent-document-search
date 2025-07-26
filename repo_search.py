import os

def search_string_in_repo(repo_path, search_string):
    matching_files = []

    for root, dirs, files in os.walk(repo_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                # Attempt to read the file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if search_string.lower() in content.lower(): # Case-insensitive search
                        matching_files.append(filename)
            except (UnicodeDecodeError, PermissionError, FileNotFoundError):
                # Skip files that can't be read (e.g., binary files, permission issues)
                continue
            except Exception as e:
                # Catch any other unexpected errors during file processing
                print(f"Error processing file {file_path}: {e}")
                continue

    # Return unique filenames
    return list(set(matching_files))

if __name__ == "__main__":
    # This block runs only when repo_search.py is executed directly
    repo_path = input("Enter the path to the repository folder: ").strip()
    search_string = input("Enter the string to search: ").strip()

    result = search_string_in_repo(repo_path, search_string)

    if result:
        print(f"\nFiles containing the string '{search_string}':")
        for filename in result:
            print(f"- {filename}")
    else:
        print(f"\nNo files found containing the string '{search_string}'.")
