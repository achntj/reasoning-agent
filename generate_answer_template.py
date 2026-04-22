#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations
from reasoning_agent import answer_question #import to call LLM and get the answer string for each question

import json
from pathlib import Path
from typing import Any, Dict, List


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions[:1], start=1):
        # Example: assume you have an agent loop that produces an answer string.
        # real_answer = agent_loop(question["input"])
        # answers.append({"output": real_answer})
        answer = answer_question(question["input"]) #Replaced the logic inside build_answers() to call agent loop and return the final answer string for each question
        answers.append({"output": answer})
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


# def main() -> None:
#     questions = load_questions(INPUT_PATH)
#     answers = build_answers(questions)

#     with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
#         json.dump(answers, fp, ensure_ascii=False, indent=2)

#     with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
#         saved_answers = json.load(fp)
#     validate_results(questions, saved_answers)
#     print(
#         f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
#         "and validated format successfully."
#     )

# temporary main function that skips validation of file lengths to test api calls and answer generation. 
# Remember to switch back to the original main function when running test on whole file
def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    print(f"Wrote {len(answers)} answers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

