from pathlib import Path
import subprocess

module_directory = Path(__file__).parent
data_file_directory = module_directory / ".data_files"
output_directory = module_directory.parent / "output"

def get_git_revision_short_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        # If git isn't available (sometimes the case in tests), return a blank
        return ""
