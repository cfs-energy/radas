import subprocess

def get_git_revision_short_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        # If git isn't available (sometimes the case in tests), return a blank
        return ""
