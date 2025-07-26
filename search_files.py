import os
from dotenv import load_dotenv

# Load environment variables from .env file.
# Note: In this integrated setup, app.py loads dotenv,
# but keeping it here makes this script runnable independently for testing.
load_dotenv()

def list_files_in_folder(folder_path):
    """Returns a list of files in the given folder."""
    try:
        # Ensure the path exists and is a directory
        if not os.path.isdir(folder_path):
            print(f"Error: Folder '{folder_path}' is not a valid directory.")
            return []
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied to access '{folder_path}'.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while listing files in '{folder_path}': {e}")
        return []

def calculate_suitability_score(file_name, search_text):
    """Calculates a score based on how well the filename matches the search text."""
    file_name_lower = file_name.lower()
    search_text_lower = search_text.lower()

    score = 0
    occurrence = 0

    if file_name_lower == search_text_lower + ".txt": # Exact match with .txt extension
        score = 100
        occurrence = 1
    elif file_name_lower.startswith(search_text_lower): # Filename starts with search text
        score = 95
        occurrence = 1
    elif search_text_lower in file_name_lower: # Search text is anywhere in the filename
        score = 90 + file_name_lower.count(search_text_lower) * 5 # Bonus for multiple occurrences
        occurrence = file_name_lower.count(search_text_lower)
    else:
        score = 0
        occurrence = 0

    return score, occurrence

if __name__ == "__main__":
    # This block runs only when search_files.py is executed directly
    # Get the folder path from environment variable
    folder_path = os.getenv('REPO_PATH') # Use REPO_PATH as defined in .env

    if not folder_path:
        print("Error: REPO_PATH is not set in the .env file.")
    else:
        search_text = input("Enter text to search for in filenames: ").strip().lower()

        files = list_files_in_folder(folder_path)

        if not files:
            print("\nNo files found in the specified folder or path is invalid.")
        else:
            print(f"\nSearching filenames in '{folder_path}' for: '{search_text}'\n")

            file_scores = {file: calculate_suitability_score(file, search_text) for file in files}

            # Sort by score (desc), then occurrences (desc), then filename (asc)
            sorted_files = sorted(file_scores.items(), key=lambda x: (-x[1][0], -x[1][1], x[0]))

            print(" **Sorted by Suitability & Occurrences:**\n")
            found_any = False
            for file, (score, occurrences) in sorted_files:
                if score > 0: # Only show relevant files
                    print(f"- {file} | Score: {score} | Occurrences: {occurrences}")
                    found_any = True
            if not found_any:
                print(f"No files found matching '{search_text}' with a positive suitability score.")
