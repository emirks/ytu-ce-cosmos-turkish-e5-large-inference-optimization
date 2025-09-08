# Auto-Commit Script Usage

This script automatically commits and pushes each file individually with AI-generated commit messages.

## Prerequisites

1. Install required dependencies:
   ```bash
   pip install openai
   ```

2. Set your OpenAI API key as an environment variable:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the script from the repository root:
   ```bash
   python auto_commit_push.py
   ```

2. The script will:
   - Detect all untracked files
   - Show you a list of files to be processed
   - Ask for confirmation
   - For each file:
     - Generate an AI commit message based on file content
     - Add, commit, and push the file individually

## Features

- **AI-generated commit messages**: Uses OpenAI GPT-3.5 to generate meaningful commit messages
- **Individual commits**: Each file gets its own commit for better history
- **Large file filtering**: Automatically skips files larger than 100MB
- **Error handling**: Continues processing even if individual files fail
- **Rate limiting**: Small delays between operations to avoid API limits

## Repository Configuration

The script is configured to push to:
`https://github.com/emirks/ytu-ce-cosmos-turkish-e5-large-inference-optimization.git`

## Example Output

```
--- Processing: embedding_model.py ---
Generating commit message...
Commit message: [feat] Add Turkish embedding model optimization
Added embedding_model.py
Committed embedding_model.py
Pushed embedding_model.py
``` 