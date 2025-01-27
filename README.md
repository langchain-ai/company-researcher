# Company Researcher Agent

Company Researcher Agent searches the web for information about a user-supplied company and returns it in a structured format defined by user-supplied JSON schema.

## 🚀 Quickstart with LangGraph server

Set API keys for the LLM of choice (Anthropic is set by default in `src/agent/graph.py`) and [Tavily API](https://tavily.com/):
```
cp .env.example .env
```

Clone the repository and launch the assistant [using the LangGraph server](https://langchain-ai.github.io/langgraph/cloud/reference/cli/#dev):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/langchain-ai/company-researcher.git
cd company-researcher
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

![company_people_researcher](https://github.com/user-attachments/assets/f651d18c-8cf8-4dde-87cb-3daed59c7fa0)

## Run as REST API

### 🐳 with Docker

Clone the repository and launch using Docker Compose:
```bash
git clone https://github.com/langchain-ai/company-researcher.git
cd company-researcher
cp .env.example .env
docker compose up
```

### 🚀 with Uvicorn

Clone and run directly with uvicorn:
```bash
git clone https://github.com/langchain-ai/company-researcher.git
cd company-researcher
cp .env.example .env
pip install .
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Test the API

Health check:
```bash
curl http://localhost:8000/health
```

Research company:
```bash
curl -X POST \
  http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"company": "openai", "user_notes": "gpt5 news"}'
```

## How it works

Company Researcher Agent follows a multi-step research and extraction workflow that separates web research from schema extraction, allowing for better resource management and comprehensive data collection:

   - **Research Phase**: The system performs intelligent web research on the input company:
     - Uses an LLM to generate targeted search queries based on the schema requirements (up to `max_search_queries`)
     - Executes concurrent web searches via [Tavily API](https://tavily.com/), retrieving up to `max_search_results` results per query
     - Takes structured research notes focused on schema-relevant information
   - **Extraction Phase**: After research is complete, the system:
     - Consolidates all research notes
     - Uses an LLM to extract and format the information according to the user-defined schema
     - Returns the structured data in the exact format requested
   - **Reflection Phase**: The system evaluates the quality of extracted information:
     - Analyzes completeness of required fields
     - Identifies any missing or incomplete information
     - Generates targeted follow-up search queries if needed
     - Continues research until information is satisfactory or max reflection steps reached

## Configuration

The configuration for Company Researcher Agent is defined in the `src/agent/configuration.py` file: 
* `max_search_queries`: int = 3 # Max search queries per company
* `max_search_results`: int = 3 # Max search results per query
* `max_reflection_steps`: int = 1 # Max reflection steps

## Inputs 

The user inputs are: 

```
* company: str - A company to research
* extraction_schema: Optional[dict] - A JSON schema for the output
* user_notes: Optional[str] - Any additional notes about the company from the user
```

If a schema is not provided, the system will use a default schema (`DEFAULT_EXTRACTION_SCHEMA`) defined in `src/agent/state.py`.

### Schemas  

> ⚠️ **WARNING:** JSON schemas require `title` and `description` fields for [extraction](https://python.langchain.com/docs/how_to/structured_output/#typeddict-or-json-schema).
> ⚠️ **WARNING:** Avoid JSON objects with nesting; LLMs have challenges performing structured extraction from nested objects. See examples below that we have tested. 

Here is an example schema that can be supplied to research a company:  

* See the trace [here](https://smith.langchain.com/public/9f51fb8b-9486-4cd2-90ed-895f7932304e/r).

    <details>
    <summary>Example schema</summary>

    ```
    {
        "title": "CompanyInfo",
        "description": "Basic information about a company",
        "type": "object",
        "properties": {
            "company_name": {
                "type": "string",
                "description": "Official name of the company"
            },
            "founding_year": {
                "type": "integer",
                "description": "Year the company was founded"
            },
            "founder_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of the founding team members"
            },
            "product_description": {
                "type": "string",
                "description": "Brief description of the company's main product or service"
            },
            "funding_summary": {
                "type": "string",
                "description": "Summary of the company's funding history"
            }
        },
        "required": ["company_name"]
    }
    ```
    </details>

Here is an example of a more complex schema: 

* See the reflections steps in the trace [here](https://smith.langchain.com/public/36f0d917-4edd-4d55-8dbf-6d6ec8a25754/r).

    <details>
    <summary>Example complex schema</summary>

    ```
    HARD_EXTRACTION_SCHEMA = {
        "title": "CompanyInfo",
        "description": "Comprehensive information about a company with confidence tracking",
        "type": "object",
        "properties": {
            "company_name": {
                "type": "string",
                "description": "Official name of the company"
            },
            "verified_company": {
                "type": "boolean",
                "description": "Confirmation this is the intended company, not a similarly named one"
            },
            "similar_companies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of similarly named companies that could be confused with the target"
            },
            "distinguishing_features": {
                "type": "string",
                "description": "Key features that distinguish this company from similarly named ones"
            },
            "key_executives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "verification_date": {"type": "string"},
                        "confidence_level": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "uncertain"]
                        },
                        "source": {"type": "string"}
                    }
                }
            },
            "org_chart_summary": {
                "type": "string",
                "description": "Brief description of organizational structure"
            },
            "leadership_caveats": {
                "type": "string",
                "description": "Any uncertainties or caveats about leadership information"
            },
            "main_products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "launch_date": {"type": "string"},
                        "current_status": {"type": "string"}
                    }
                }
            },
            "services": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "target_market": {"type": "string"}
                    }
                }
            },
            "recent_developments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "source_url": {"type": "string"},
                        "significance": {"type": "string"}
                    }
                },
                "description": "Major news and developments from the last 6 months"
            },
            "historical_challenges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_type": {"type": "string"},
                        "description": {"type": "string"},
                        "date_period": {"type": "string"},
                        "resolution": {"type": "string"},
                        "current_status": {"type": "string"}
                    }
                },
                "description": "Past challenges, issues, or controversies"
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "date_accessed": {"type": "string"},
                        "information_type": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of information sourced from this link (e.g., leadership, products, news)"
                        }
                    }
                }
            },
            "company_summary": {
                "type": "string",
                "description": "Concise, dense summary of the most important company information (max 250 words)"
            }
        },
        "required": [
            "company_name",
            "verified_company",
            "company_summary",
            "key_executives",
            "main_products",
            "sources"
        ]
    }
    ```
    </details>


## Evaluation

Prior to engaging in any optimization, it is important to establish a baseline performance. This repository includes:

1. A dataset consisting of a list of companies and the expected structured information to be extracted for each company.
2. An evaluation script that can be used to evaluate the agent on this dataset.

### Set up

Make sure you have the LangSmith CLI installed:

```shell
pip install langsmith
```

And set your API key:

```shell
export LANGSMITH_API_KEY=<your_langsmith_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

### Evaluation metric

A score between 0 and 1 is assigned to each extraction result by an LLM model that acts
as a judge.

The model assigns the score based on how closely the extracted information matches the expected information.

### Get the dataset

Create a new dataset in LangSmith using the code in the `eval` folder:

```shell
python eval/create_dataset.py
```

### Run the evaluation

To run the evaluation, you can use the `run_eval.py` script in the `eval` folder. This will create a new experiment in LangSmith for the dataset you created in the previous step.

```shell
python eval/run_eval.py --experiment-prefix "My custom prefix" --agent-url http://localhost:2024
```
