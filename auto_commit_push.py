#!/usr/bin/env python3
"""
Auto-commit and push script that commits each file individually with AI-generated commit messages.
"""

import os
import subprocess
import sys
from pathlib import Path
import openai
from typing import List, Optional
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
REPO_URL = "https://github.com/emirks/ytu-ce-cosmos-turkish-e5-large-inference-optimization.git"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            check=True,
            shell=True if os.name == 'nt' else False
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        raise

def get_untracked_files() -> List[str]:
    """Get list of untracked files from git status."""
    result = run_command(["git", "status", "--porcelain"])
    files = []
    
    for line in result.stdout.strip().split('\n'):
        if line.startswith('??'):
            file_path = line[3:].strip()
            # Skip directories and large files
            if not file_path.endswith('/') and not is_large_file(file_path):
                files.append(file_path)
    
    return files

def is_large_file(file_path: str) -> bool:
    """Check if file is too large (>100MB) to commit."""
    try:
        file_size = os.path.getsize(file_path)
        return file_size > 100 * 1024 * 1024  # 100MB
    except OSError:
        return False

def get_file_content_preview(file_path: str, max_lines: int = 20) -> str:
    """Get a preview of file content for context."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append("... (truncated)")
                    break
                lines.append(line.rstrip())
            return '\n'.join(lines)
    except Exception as e:
        return f"Could not read file: {e}"

def generate_commit_message(file_path: str) -> str:
    """Generate a commit message using OpenAI API."""
    file_content = get_file_content_preview(file_path)
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1]
    
    prompt = f"""Generate a concise git commit message for this file. The message should be in the format: "[type] Brief description"

File: {file_name}
Extension: {file_ext}
Content preview:
{file_content}

Rules:
- Use conventional commit types: feat, fix, docs, style, refactor, test, chore
- Keep message under 50 characters
- Be specific about what the file does
- Examples: "[feat] Add embedding optimization script", "[docs] Add Pinecone setup guide"

Commit message:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise git commit messages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        message = response.choices[0].message.content.strip()
        # Clean up the message
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]
        
        return message
        
    except Exception as e:
        print(f"Error generating commit message for {file_path}: {e}")
        # Fallback to simple message
        return f"[chore] Add {file_name}"

def setup_remote():
    """Set up the remote repository."""
    try:
        # Check if remote exists
        result = run_command(["git", "remote", "get-url", "origin"])
        print(f"Remote already configured: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        # Add remote
        print("Setting up remote repository...")
        run_command(["git", "remote", "add", "origin", REPO_URL])
        print(f"Added remote: {REPO_URL}")

def commit_and_push_file(file_path: str):
    """Commit and push a single file."""
    print(f"\n--- Processing: {file_path} ---")
    
    # Generate commit message
    print("Generating commit message...")
    commit_message = generate_commit_message(file_path)
    print(f"Commit message: {commit_message}")
    
    try:
        # Add file
        run_command(["git", "add", file_path])
        print(f"Added {file_path}")
        
        # Commit file
        run_command(["git", "commit", "-m", commit_message])
        print(f"Committed {file_path}")
        
        # Push to remote
        run_command(["git", "push", "origin", "main"])
        print(f"Pushed {file_path}")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")
        # Continue with next file
        return False
    
    return True

def main():
    """Main function to process all files."""
    print("Auto-commit and push script starting...")
    
    # Setup remote
    setup_remote()
    
    # Get untracked files
    files = get_untracked_files()
    
    if not files:
        print("No untracked files found.")
        return
    
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {f}")
    
    # Ask for confirmation
    response = input(f"\nProceed with committing and pushing {len(files)} files? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in files:
        if commit_and_push_file(file_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(files)}")

if __name__ == "__main__":
    main() 