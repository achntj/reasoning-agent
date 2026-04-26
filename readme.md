# CSE 476 Final Project 

## Overview

A general-purpose reasoning agent that solves arbitrary question-answering tasks using 9 inference-time techniques. 

## Requirements

- Python 3.10+
- Access the SOL API and ASU network(either directly or through VPN)
- An API key from the ASU Voyager (provided for easier grading)

## Installation

```bash
pip install -r requirements.txt
```



## Running the Agent

To generate answers for the test data:

```bash
python generate_answer_template.py
```

This reads `cse_476_final_project_test_data.json` and writes `cse_476_final_project_answers.json`.


## Project Structure

```
reasoning_agent.py            
generate_answer_template.py  
README.md
cse_476_final_project_answers.json     #NOTE: Contains only first 100 answers due to time contraints 
.env                          
```

## Techniques Implemented

1. Direct zero-shot prompting
2. Chain-of-Thought (CoT)
3. Decomposition
4. Self-Refine
5. Option Elimination (MCQ)
6. Math Verification pass
7. ReACT-style unit normalisation
8. Self-Consistency
9. Translation pre-processing

## Notes

- Although .env file and API key is usually not provided, it has been here for easier grading
- cse_476_final_project_answers.json contains first 100 answers due to running the agent on full test data took much longer than anticipated, perhaps due to high volume of requests to LLM by entire class
- Maximum 8 LLM calls per question
- Uses qwen3-30b-a3b-instruct-2507 via the ASU SOL API
- No external paid APIs are used