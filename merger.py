import os

def merge_python_files(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.py'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(f"\n\n### START OF {file_name} ###\n\n")
                    outfile.write(infile.read())
                    outfile.write(f"\n\n### END OF {file_name} ###\n\n")

folder_path = "Streamlit"  # Change to your desired folder path
output_file = "merged_output.txt"  # Output file name

merge_python_files(folder_path, output_file)
print(f"Merged file saved as {output_file}")
