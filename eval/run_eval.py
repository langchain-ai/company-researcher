import argparse
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResults
from pydantic import BaseModel, Field

from langgraph.pregel.remote import RemoteGraph


client = Client()

NUMERIC_FIELDS = (
    "total_funding_mm_usd",
    "latest_round_amount_mm_usd",
)
EXACT_MATCH_FIELDS = (
    "website",
    "crunchbase_profile",
    "headquarters",
    "year_founded",
    "latest_round",
    "latest_round_date",
)
FUZZY_MATCH_FIELDS = ("name", "ceo", "description")

DEFAULT_DATASET_NAME = "Startup Data Enrichment"
DEFAULT_GRAPH_ID = "company_researcher"
DEFAULT_AGENT_URL = "http://localhost:2024"

judge_llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

EVALUATION_PROMPT = f"""You are an evaluator tasked with assessing the accuracy of an agent's output compared to the expected output. Follow these instructions:

1. **Numeric Fields Evaluation**: For fields {NUMERIC_FIELDS}, check if the agent's output is within 10% of the expected value. Score 1 if yes, 0 if no.
2. **Exact Match Evaluation**: For fields {EXACT_MATCH_FIELDS}, check if the agent's output matches the expected output EXACTLY. Score 1 if yes, 0 if no.
3. **Fuzzy Match Evaluation**: For fields {FUZZY_MATCH_FIELDS}, check if the agent's output matches the expected output APPROXIMATELY. Score 1 if yes, 0 if no.
4. **Overall Evaluation**: Return final score that is a fraction of fields that have score of 1. For example, if 1/5 fields has score of 1, the final score is 0.2."""


def evaluate_agent(outputs: dict, reference_outputs: dict):
    if "info" not in outputs:
        raise ValueError("Agent output must contain 'info' key")

    class Score(BaseModel):
        """Evaluate the agent's output against the expected output."""

        reason: str = Field(
            description="A brief explanation for why you scored the agent's output as you did."
        )
        score: float = Field(
            description="A score between 0 and 1 indicating the accuracy of the agent's output compared to the expected output. 1 is a perfect match."
        )

    score = judge_llm.with_structured_output(Score).invoke(
        [
            {
                "role": "system",
                "content": EVALUATION_PROMPT,
            },
            {
                "role": "user",
                "content": f'Actual output: {outputs["info"]}\nExpected output: {reference_outputs["info"]}',
            },
        ]
    )
    return score.score


# PUBLIC API


def transform_dataset_inputs(inputs: dict) -> dict:
    """Transform LangSmith dataset inputs to match the agent's input schema before invoking the agent."""
    # see the `Example input` in the README for reference on what `inputs` dict should look like
    # the dataset inputs already match the agent's input schema, but you can add any additional processing here
    return inputs


def transform_agent_outputs(outputs: dict) -> dict:
    """Transform agent outputs to match the LangSmith dataset output schema."""
    # see the `Example output` in the README for reference on what the output should look like
    return {"info": outputs["info"]}


def make_agent_runner(graph_id: str, agent_url: str):
    """Wrapper that transforms inputs/outputs to match the expected eval schema and invokes the agent."""
    agent_graph = RemoteGraph(graph_id, url=agent_url)

    def run_agent(inputs: dict) -> dict:
        """Run the agent on the inputs from the LangSmith dataset record, return outputs conforming to the LangSmith dataset output schema."""
        transformed_inputs = transform_dataset_inputs(inputs)
        response = agent_graph.invoke(transformed_inputs)
        return transform_agent_outputs(response)

    return run_agent


def run_eval(
    *,
    dataset_name: str,
    graph_id: str = DEFAULT_GRAPH_ID,
    agent_url: str = DEFAULT_AGENT_URL,
    experiment_prefix: Optional[str] = None,
) -> EvaluationResults:
    dataset = client.read_dataset(dataset_name=dataset_name)
    run_agent = make_agent_runner(graph_id, agent_url)
    results = evaluate(
        run_agent,
        data=dataset,
        evaluators=[evaluate_agent],
        experiment_prefix=experiment_prefix,
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name of the dataset to evaluate against",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=DEFAULT_GRAPH_ID,
        help="ID of the graph to evaluate",
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default=DEFAULT_AGENT_URL,
        help="URL of the deployed agent to evaluate",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        help="Experiment prefix for the evaluation",
    )
    args = parser.parse_args()

    run_eval(
        dataset_name=args.dataset_name,
        graph_id=args.graph_id,
        agent_url=args.agent_url,
        experiment_prefix=args.experiment_prefix,
    )
