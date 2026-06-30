# Testing Strategy

## Unit Tests (per node)
Each node in nodes.py has an isolated test in tests/test_nodes.py.
Mock the LLM with a fixture that returns canned responses.

## Integration Test: Full Graph Run
tests/test_graph.py runs the full graph with a seeded ChromaDB.

## Hallucination Test Cases
Maintain a golden test set in tests/fixtures/hallucination_cases.json:
[
  {
    "question": "What is the CEO's name?",
    "documents": ["The company was founded in 2010."],
    "expected_verdict": "insufficient"
  }
]

## Running Tests
pytest tests/ -v
pytest tests/test_critic.py -v   # critic-only