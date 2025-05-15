# Talk2Table

A modular Python project that processes user queries through a Streamlit interface using custom agents and data pipelines.

## Features

- Modular design: Components for data preparation, query parsing, and agent tools
- Natural language query processing
- Streamlit frontend for interactive querying
- Customizable agents and tools

## Project Structure

```plaintext
.
├── app.py                  # Streamlit app
├── main.py                # Main Python runner (optional backend/testing)
└── modules/
    ├── agent_tools.py      # Tools used by agents to process or answer queries
    ├── data_preparation.py # Dataset ingestion and vectorization
    └── query_processing.py # Main query parsing and processing logic
```

## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app using Streamlit:

```bash
streamlit run app.py
```

## Modules Overview

- modules/agent_tools.py  
  → Custom tools and functions used by agents to enhance or retrieve responses.

- modules/data_preparation.py  
  → Handles loading, cleaning, and embedding of data.

- modules/query_processing.py  
  → Interprets and routes user queries to the right logic or agent.

## Requirements

- Python ≥ 3.8
- streamlit
- pandas, numpy
- sentence-transformers or openai (if vector search is used)
- any other libraries used inside your modules

(Add a requirements.txt file or install with pip freeze > requirements.txt)

## Extending

You can add new tools or data sources by extending:
- agent_tools.py — to support new response functions
- data_preparation.py — to preprocess different types of content
- query_processing.py — to introduce new query intent recognition logic

