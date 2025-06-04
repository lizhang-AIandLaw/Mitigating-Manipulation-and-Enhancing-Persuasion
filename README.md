# Reflective Multi-Agent Argumentation

## 1. Overview

This script is designed to conduct experiments on legal argumentation using various Large Language Model (LLM) agent architectures. It focuses on trade secret misappropriation claims, where AI agents generate 3-ply arguments (Plaintiff's Argument, Defendant's Counterargument, Plaintiff's Rebuttal) based on provided case factors and precedent cases (TSCs). The script automates scenario generation, argument generation by different agent setups, and evaluation of these arguments based on several metrics.

## 2. Features

-   **Scenario Generation**: Dynamically creates legal scenarios with varying complexities and arguability (non-arguable, mismatched, arguable).
-   **Multiple Agent Architectures**:
    *   **Single Agent**: A single LLM call to generate the full 3-ply argument.
    *   **Prompt-Enhanced Single Agent**: Single agent with specific prompt injections for tasks like precedent identification or analogy/distinction.
    *   **Multi-Agent Debate**: Separate LLM calls for Plaintiff and Defendant, simulating a debate flow.
    *   **Multi-Agent Debate with Reflection**: A more advanced setup where arguments are reviewed and potentially revised by Factor Analyst and Argument Polisher agents before finalization.
-   **LLM Integration**: Supports OpenAI (e.g., GPT models) and Groq API clients.
-   **Factor Distillation**: Extracts legal factors mentioned in the generated arguments for evaluation.
-   **Automated Evaluation Metrics**:
    *   **Precedent Identification Accuracy**: Measures how accurately the agent identifies and uses relevant precedent cases.
    *   **Hallucination Accuracy**: Assesses the extent to which the agent introduces fabricated case factors.
    *   **Factor Utilization Recall**: Measures how well the agent utilizes the actual provided case factors.
    *   **Successful Abstention Rate**: For non-arguable scenarios, this checks if the agent correctly abstains from making an argument.
-   **Comprehensive Logging**: Outputs detailed experiment data in JSONL format, including generated arguments, distilled factors, and all calculated metrics.
-   **Command-Line Interface**: Allows configuration of model types, model names, number of scenarios, complexity, and output format.

## 3. Prerequisites

-   Python 3.x
-   Access to OpenAI API and/or Groq API.
-   Environment variables set for API keys.

## 4. Setup

1.  **Clone the repository (if applicable) or ensure all script files are in the same directory.**
    The script relies on `Scenario_Generator.py` being accessible.

2.  **Install dependencies**:
    The script requires the `openai` and `groq` Python libraries, and `python-dotenv` for managing environment variables.
    You can install them using pip:
    ```bash
    pip install openai groq python-dotenv
    ```

3.  **Create a `.env` file**:
    In the same directory as the script, create a file named `.env` and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    GROQ_API_KEY="your_groq_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` and `"your_groq_api_key_here"` with your actual API keys.

## 5. Usage

Run the script from the command line.

```bash
python multi_agent_factor.py [OPTIONS]
```

### Command-Line Arguments:

-   `--model_type`: (Optional) LLM client type. Choices: `openai`, `groq`. Default: `openai`.
-   `--model_name`: (Optional) Name of the LLM to use for arguments. Default: `gpt-4o-mini`.
-   `--reflection_model_name`: (Optional) Name of LLM for reflection agents (Factor Analyst, Polisher). Default: `gpt-4o`.
-   `--distiller_model_name`: (Optional) Name of LLM for Factor Distiller agent. Default: `gpt-4o`.
-   `--distiller_client_type`: (Optional) LLM client type for the distiller model. Choices: `openai`, `groq`. Default: `openai`.
-   `--num_scenarios_per_type`: (Optional) Number of scenarios to generate per type (non-arguable, mismatched, arguable). Min 1. Default: `1`.
-   `--complexity`: (Optional) Complexity for scenario generation (number of factors). Min 1. Default: `3`.
-   `--output_format`: (Optional) Output format for scenario generation. Choices: `factor`, `code`. Default: `factor`.
-   `--log_prefix`: (Optional) Prefix for the experiment log file name. Default: `multi_agent_factor_experiment`.

### Example:

To run an experiment with Groq's Llama3 70b model for argument generation, using 2 scenarios per type with complexity 5:

```bash
python multi_agent_factor.py --model_type groq --model_name llama3-70b-8192 --num_scenarios_per_type 2 --complexity 5 --log_prefix my_llama_test
```

## 6. Key Components

### a. Scenario Generation (`Scenario_Generator.py`)
Responsible for creating the input scenarios, including:
-   Input Case Factors
-   TSC1 (Trade Secret Case 1 for Plaintiff) Factors and Outcome
-   TSC2 (Trade Secret Case 2 for Defendant) Factors and Outcome
It can generate `arguable`, `non-arguable`, and `mismatched` scenarios.

### b. Agent Prompts
The script defines several system prompts for different AI agents:
-   `BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT`: Instructs the main argument generation LLM on how to construct the 3-ply argument and the expected JSON output format.
-   `PROMPT_INJECTION_TASK1_PRECEDENT_ID`: Additional instructions for the "precedent identification" focused agent, emphasizing when to terminate if no suitable precedent is found.
-   `PROMPT_INJECTION_TASK2_3_ANALOGY_DISTINCTION`: Additional instructions for the "analogy/distinction" focused agent, guiding it on how to use shared and distinguishing factors.
-   `FACTOR_ANALYST_SYSTEM_PROMPT`: Instructs the Factor Analyst agent on how to analyze an argument segment against case factors, identifying discrepancies.
-   `ARGUMENT_POLISHER_SYSTEM_PROMPT`: Instructs the Argument Polisher agent on how to review the argument and Factor Analyst's report, providing feedback and revision instructions.
-   `FACTOR_DISTILLER_SYSTEM_PROMPT`: Instructs the Factor Distiller agent to extract all unique legal factors mentioned in a generated argument.

### c. LLM Interaction (`get_llm_response`)
A centralized function to interact with the chosen LLM API (OpenAI or Groq). It handles API calls, response parsing (including JSON extraction), and error handling.

### d. Agent Architectures
-   `run_single_agent`: Generates the entire 3-ply argument in one go.
-   `run_prompt_enhanced_single_agent`: Similar to `run_single_agent` but with additional prompt injections.
-   `run_multi_agent_debate`: Simulates a debate by generating each part of the 3-ply argument sequentially, with each agent aware of the previous arguments.
-   `run_multi_agent_debate_with_reflection`: An advanced architecture where each argument ply is generated and then critically reviewed by Factor Analyst and Argument Polisher agents. If issues are found, the Argument Developer revises its argument based on the feedback. This process can iterate up to `REFLECTION_THRESHOLD` times.

### e. Evaluation Metrics
-   `calculate_precedent_identification_accuracy`: Checks if the cited precedents are appropriate (correct outcome for the side citing it) and if the claimed common factors are actually present.
-   `calculate_hallucination_accuracy_and_factor_recall`:
    -   **Hallucination Accuracy**: Measures the proportion of claimed factors that are *not* present in the ground truth for the respective case (Input, TSC1, TSC2). Higher is better.
    -   **Factor Utilization Recall**: Measures the proportion of actual ground truth factors (across all cases) that were mentioned in the generated argument. Higher is better.
-   `calculate_successful_abstention`: For scenarios designed to be non-arguable, this checks if the agent correctly abstained from generating a substantive argument.

## 7. Logging

-   Experiment results are logged to a JSONL file (one JSON object per line, per experiment run).
-   The log file is named using the provided `--log_prefix` and a timestamp (e.g., `multi_agent_factor_experiment_YYYYMMDD_HHMMSS.jsonl`).
-   Each log entry contains:
    -   Scenario details (type, ID, complexity, ground truth text and parsed structure).
    -   Model configuration (type, names).
    -   Architecture used.
    -   Generation time.
    -   The full generated argument (JSON or text).
    -   Distilled factors from the argument.
    -   All calculated evaluation metrics for that run.

This allows for detailed post-experiment analysis.
