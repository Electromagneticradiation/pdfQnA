# READOPHILE
PDF QnA tool (BYTE ML task)
This project uses ChromaDB + Jina v4 embeddings + Groq llama8b with a Streamlit interface.

## Prerequisites
- Python 3.13
- `pip` or `pdm` (package managers)

## Installation

1. Clone the repo
   ``` bash
   git clone <your-repo-url>
   cd <your-repo-name> ```

2. Setup the environment 

2.1 If you have pip : ```
    python -m venv venv
    source venv/bin/activate      # Linux / Mac
    venv\Scripts\activate         # Windows 
    pip install -r requirements.txt ```

2.2 Or use pdm : ```
  powershell -ExecutionPolicy ByPass -c "irm https://pdm-project.org/install-pdm.py | py -"  # Windows
  curl -sSL https://pdm-project.org/install-pdm.py | python3 -                               # Mac / Linux
  pdm install ```
  
3. Set up API keys

Create a .env file in the project root
Write : ```
JINA_API_KEY = "your_jina_api_key"  # get yours from https://jina.ai/api-dashboard/key-manager
GROQ_API_KEY = "your_groq_api_key"  # from https://console.groq.com/keys ```

4. Run the Streamlit app ```
streamlit run main.py             # pip
pdm run streamlit run main.py     # pdm ```

This will start a local server (http://localhost:8501)
