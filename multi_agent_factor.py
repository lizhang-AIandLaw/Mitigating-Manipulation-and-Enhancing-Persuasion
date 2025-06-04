import os
import json
import csv
import datetime
import time
import argparse
import re
import random
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

# Assuming Scenario_Generator.py is in the same directory or accessible in PYTHONPATH
from scenario_generator import generate_datasets, ScenarioGenerator

# Load environment variables from .env file
load_dotenv()

# --- API Client Setup ---
def get_openai_client():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
    return OpenAI(api_key=openai_api_key)

def get_groq_client():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
    return Groq(api_key=groq_api_key)

OPENAI_CLIENT = get_openai_client()
GROQ_CLIENT = get_groq_client()

# --- Constants ---
REFLECTION_THRESHOLD = 1
DEFAULT_MAX_TOKENS = 1000 # As per paper
DEFAULT_TEMPERATURE = 0.0 # As per paper
DEFAULT_TOP_P = 1.0 # As per paper
DEFAULT_FREQUENCY_PENALTY = 0.0 # As per paper
DEFAULT_PRESENCE_PENALTY = 0.0 # As per paper

# --- Prompts ---
BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT = """
You are an AI assistant tasked with formulating legal arguments for trade secret misappropriation claims.
Construct a 3-Ply Argument:
1. Plaintiff's Argument: Cite a relevant Trade Secret Case (TSC1) with a favorable outcome for Plaintiff. Highlight shared factors between the input case and TSC1.
2. Defendant's Counterargument: Distinguish TSC1. Cite a counterexample (TSC2, with a Defendant-favorable outcome) and draw an analogy to the input case, highlighting shared factors.
3. Plaintiff's Rebuttal: Address and distinguish TSC2, reinforcing the Plaintiff's original argument.

Base your arguments on the provided factors. Ensure logical consistency.
Output the 3-ply argument in a single JSON object with keys: "Plaintiff's Argument", "Defendant's Counterargument", "Plaintiff's Rebuttal".

Example Input Case:
F1 Disclosure-in-negotiations (D)
F4 Agreed-not-to-disclose (P)
F6 Security-measures (P)

Example TSC1 (for Plaintiff):
outcome Plaintiff
F4 Agreed-not-to-disclose (P)
F6 Security-measures (P)
F7 Brought-tools (P)

Example TSC2 (for Defendant):
outcome Defendant
F1 Disclosure-in-negotiations (D)
F5 Agreement-not-specific (D)

Example JSON Output:
{
  "Plaintiff's Argument": "Factors F4 Agreed-not-to-disclose (P) and F6 Security-measures (P) were present in both the input case and TSC1 (outcome Plaintiff), supporting the Plaintiff. The input case also has F12...",
  "Defendant's Counterargument": "TSC1 is distinguishable because it had F7 Brought-tools (P), not in the input case. Furthermore, the input case has F1 Disclosure-in-negotiations (D). TSC2 (outcome Defendant) is analogous; both share F1 Disclosure-in-negotiations (D) and F5 Agreement-not-specific (D).",
  "Plaintiff's Rebuttal": "TSC2 is distinguishable as the input case lacks F5 Agreement-not-specific (D) and has strong pro-plaintiff factors like F4 and F6 not in TSC2."
}
If you cannot make a valid argument for a step (e.g., no suitable precedent), state that clearly for that part of the argument.
"""

PROMPT_INJECTION_TASK1_PRECEDENT_ID = """
Crucially, if no common factors are found between the input case and a cited TSC, OR if the cited TSC's outcome does not favor your side for that specific ply of the argument, you MUST state 'TERMINATE: No suitable precedent found for this argument.' at the beginning of that specific argument part and do not generate further for that part.
Example for Plaintiff's Argument: If TSC1 (outcome Defendant) is provided for Plaintiff, or no common factors exist, Plaintiff's Argument should start with 'TERMINATE: ...'.
"""

PROMPT_INJECTION_TASK2_3_ANALOGY_DISTINCTION = """
When constructing arguments:
- For analogies (your side citing a case): Maximize the use of genuinely shared factors that support your position to strengthen the analogical reasoning.
- For distinctions (against opponent's case) and counterarguments: Highlight as many valid distinguishing factors as possible to emphasize critical differences and undermine opposing reasoning. Ensure cited counterexamples strongly support your position.
"""

COMBINED_ENHANCED_PROMPT_INJECTION = PROMPT_INJECTION_TASK1_PRECEDENT_ID + "\n\n" + PROMPT_INJECTION_TASK2_3_ANALOGY_DISTINCTION

FACTOR_ANALYST_SYSTEM_PROMPT = """
You are a Factor Analyst Agent. Your task is to analyze a given legal argument segment against an input case and relevant Trade Secret Case(s) (TSCs). Your goal is to determine if the argument segment must be abstained from, if it requires correction of factual errors, or if it appears valid based on the provided data.

**IMPORTANT CONTEXT:** You will be analyzing one ply of a 3-ply argument at a time (Plaintiff's Argument using TSC1, Defendant's Counterargument using TSC2, or Plaintiff's Rebuttal addressing TSC2 and reinforcing with TSC1). Pay close attention to which party is arguing and which TSC is primarily being cited or addressed in the segment provided.

Follow this process STRICTLY:

1.  **Identify the Context:**
    *   Determine the party making the argument segment (Plaintiff or Defendant).
    *   Identify the primary TSC being cited or addressed in this segment (TSC1 for Plaintiff's Argument, TSC2 for Defendant's Counterargument, TSC1 & TSC2 for Plaintiff's Rebuttal). You will be given the factors and outcome for the relevant TSC(s).

2.  **Determine if Abstention is REQUIRED (This is the FIRST and most critical check):**
    *   **Focus ONLY on the PRIMARY cited TSC** for the argument ply being evaluated (TSC1 for Plaintiff's Arg, TSC2 for Defendant's Counter). For Rebuttal, consider the check against TSC1 for reinforcement.
    *   Verify the actual common factors between the input case and this *primary* cited TSC based *only* on the provided factor lists. Ignore factors mentioned from other, non-primary TSCs during this step.
    *   Check the actual outcome of the *primary* cited TSC and whether it favors the arguing party (Plaintiff needs Plaintiff-outcome TSC1, Defendant needs Defendant-outcome TSC2).
    *   **Abstention IS REQUIRED** and you MUST output "REQUIRES_ABSTENTION" if EITHER of the following conditions is true for the primary cited TSC:
        a.  There are ZERO genuinely common factors between the input case and the *primary* cited TSC used for the core analogy/argument. (For Rebuttal, check this specifically for factors cited from TSC1 for reinforcement). Count common factors carefully. If the count is 0, abstention is mandatory.
        OR
        b.  The actual outcome of the *primary* cited TSC is UNFAVORABLE to the party making this argument segment (e.g., Plaintiff citing a Defendant-outcome TSC1, Defendant citing a Plaintiff-outcome TSC2).
    *   If abstention is required, set analysis_outcome to "REQUIRES_ABSTENTION", provide the reason, and proceed to output formatting. DO NOT proceed to step 3.

3.  **If Abstention is NOT Required, then Determine if Correction is Needed:**
    *   Identify all factors claimed as common or distinguishing in the argument segment. Compare these against the actual factor lists for the input case and *all* relevant TSCs (TSC1 & TSC2).
    *   Identify if the argument segment misrepresents the outcome of *any* cited TSC.
    *   **Correction IS REQUIRED** if:
        a.  The argument claims common factors that are fabricated (e.g., a factor claimed as common between Input and TSCX is not present in both's actual lists).
        b.  The argument claims distinguishing factors that are fabricated (e.g., claiming TSCX has factor Y which it doesn't, or claiming Input Case lacks factor Z which it has).
        c.  The argument misrepresents the actual outcome of a cited TSC (and this wasn't caught by the abstention rule).
    *   If correction is required (and abstention was not), your analysis_outcome is "REQUIRES_CORRECTION". List the specific errors.

4.  **If Neither Abstention nor Correction is Required:**
    *   Your analysis_outcome is "VALID_ARGUMENT".

**Special Notes for Plaintiff's Rebuttal:**
*   The Rebuttal aims to distinguish TSC2 (cited by Defendant) and reinforce the Plaintiff's case (potentially citing TSC1 again).
*   Analyze claims about TSC2: Are the claimed distinguishing factors accurate based on the actual factor lists of the Input Case and TSC2?
*   Analyze claims about TSC1 (if used for reinforcement): Are the claimed common factors accurate? Is the outcome still favorable?
*   Abstention (Rule 2a) applies if the reinforcement part *claims* common factors with TSC1 but there are actually zero *and* no valid distinction of TSC2 is made. Abstention (Rule 2b) applies if TSC1 (used for reinforcement) has an unfavorable outcome.
*   Correction (Rule 3) applies if *any* factor claims (common or distinguishing, regarding TSC1 or TSC2) are fabricated or misrepresented.

Output your analysis in JSON format as specified below. Ensure the JSON is the only output.

JSON Output Format:
{
  "analysis_outcome": "REQUIRES_ABSTENTION" / "REQUIRES_CORRECTION" / "VALID_ARGUMENT",
  "summary": "A concise explanation. If abstention, state the specific reason (unfavorable outcome OR zero common factors for the *primary* cited TSC). If correction, summarize key factual errors. If valid, confirm.",
  "abstention_details": { // Include this section ONLY if analysis_outcome is "REQUIRES_ABSTENTION"
    "reason_for_abstention": "No common factors found." / "Cited TSC outcome is unfavorable for the arguing party." / "Both: No common factors and unfavorable TSC outcome."
  },
  "correction_details": { // Include this section ONLY if analysis_outcome is "REQUIRES_CORRECTION"
    "fabricated_or_misrepresented_factors": ["Factor A (P) - claimed as common but not in TSC1's actual factors", "Factor B (D) - claimed for Input Case but not present in Input Case's actual factors"], // List factors that are incorrectly claimed. Be specific about the error.
    "misrepresented_tsc_outcome": "e.g., Argument claims TSC1 outcome is Plaintiff, but actual outcome is Defendant.", // Describe if TSC outcome is misrepresented. Omit or null if not applicable.
    "other_issues_for_correction": "Brief description of any other critical factual errors needing correction." // Omit or null if not applicable.
  }
}

Example 1 (Requires Abstention - unfavorable outcome):
Argument: Plaintiff's argument cites TSC1. Provided data: TSC1 actual outcome is 'Defendant'.
Output:
{
  "analysis_outcome": "REQUIRES_ABSTENTION",
  "summary": "The argument for Plaintiff, citing TSC1, must be abstained from. TSC1's actual outcome is 'Defendant', which does not favor the Plaintiff.",
  "abstention_details": {
    "reason_for_abstention": "Cited TSC outcome is unfavorable for the arguing party."
  }
}

Example 2 (Requires Abstention - no common factors):
Argument: Cites TSCX. Provided data: Input case factors {F1, F2}, TSCX factors {F3, F4}. (No common factors).
Output:
{
  "analysis_outcome": "REQUIRES_ABSTENTION",
  "summary": "The argument must be abstained from as there are no common factors between the input case and the cited TSCX.",
  "abstention_details": {
    "reason_for_abstention": "No common factors found."
  }
}

Example 3 (Requires Correction - fabricated factor):
Argument for Plaintiff cites TSC1. Provided data: TSC1 actual outcome 'Plaintiff'. Input case {F1, F2}, TSC1 {F1, F3}.
Argument claims: "Input case and TSC1 share F1 and F4." (F4 is fabricated as it's not in TSC1 and not common).
Output:
{
  "analysis_outcome": "REQUIRES_CORRECTION",
  "summary": "The argument requires correction. Factor F4 was claimed as common with TSC1, but F4 is not present in TSC1's actual factors.",
  "correction_details": {
    "fabricated_or_misrepresented_factors": ["F4 (claimed as common with TSC1 but not present in TSC1's actual factors)"]
  }
}

Example 4 (Valid Argument):
Argument for Plaintiff cites TSC1. Provided data: TSC1 actual outcome 'Plaintiff'. Input case {F1, F2}, TSC1 {F1, F3}.
Argument claims: "Input case and TSC1 share F1."
Output:
{
  "analysis_outcome": "VALID_ARGUMENT",
  "summary": "The argument segment appears valid. The cited TSC outcome favors the arguing party, and the claimed common factor (F1) is verified."
}
"""

ARGUMENT_POLISHER_SYSTEM_PROMPT = """
You are an Argument Polisher Agent. You will receive a generated legal argument segment, the original input case factors, relevant TSC factors, and the Factor Analyst's report.
Your tasks:
1. Review the argument segment for factual accuracy based on the provided case factors and Factor Analyst's report.
2. Check for logical coherence and persuasive strength. Specifically, assess factor utilization:
    a. Are all relevant supporting factors from the input case and cited TSC effectively used to build the analogy or argument?
    b. Are distinguishing factors (both in the cited TSC not present in the input case, and in the input case not present in the cited TSC) clearly highlighted when making distinctions or counterarguments?
    c. Are there any crucial factors from the input case or TSCs that have been overlooked and could significantly strengthen or weaken the argument?
3. Provide feedback on inaccuracies, argument strength, and specifically on factor utilization.
4. If revisions are needed, provide clear instructions to the Argument Developer Agent on what to correct or improve, with a strong focus on enhancing factor utilization.

Output your assessment in JSON format:
{
  "argument_segment_type": "Plaintiff's Argument / Defendant's Counterargument / Plaintiff's Rebuttal",
  "accuracy_assessment": "Accurate / Minor Inaccuracies / Major Inaccuracies",
  "strength_assessment": "Strong / Moderate / Weak (based on factor utilization and logic)",
  "factor_utilization_assessment": "Excellent / Good / Fair / Poor",
  "feedback_summary": "e.g., 'The argument correctly identifies shared factors but misses a key distinguishing factor in TSC1. Factor utilization could be improved by incorporating F_X from the input case.'",
  "revision_needed": true/false,
  "instructions_for_developer": "If revision_needed is true, provide concise instructions. e.g., 'Re-evaluate TSC1. While F4 is common, TSC1 also has F7 (P) which is a key distinction you missed. Input case has F10 (D) which weakens your analogy. Strengthen your argument by explicitly mentioning how F10 (D) is overcome or why TSC1 is still a good precedent despite it. Ensure all favorable factors for your side common to the input case and TSC1 are mentioned.'"
}
"""

FACTOR_DISTILLER_SYSTEM_PROMPT = """
You are a Factor Distiller Agent. Given a 3-ply legal argument in JSON format, extract all unique legal factors mentioned for the 'Input Case', 'TSC1', and 'TSC2'.
Factors are in the format like 'F1 Disclosure-in-negotiations (D)', 'F4 Agreed-not-to-disclose (P)', etc.
Output the results as a JSON object with keys "Input Case", "TSC1", and "TSC2", where each value is a list of unique factor strings.

Example Input Argument JSON:
{
  "Plaintiff's Argument": "Input case shares F4 (P) and F6 (P) with TSC1 (outcome Plaintiff). Input case also features F12 (P).",
  "Defendant's Counterargument": "TSC1 also had F7 (P), distinguishing it. Input case has F1 (D). TSC2 (outcome Defendant) is similar, both share F1 (D).",
  "Plaintiff's Rebuttal": "TSC2 is different, input case does not have F5 (D) which was in TSC2."
}

Example Output JSON:
{
  "Input Case": ["F4 Agreed-not-to-disclose (P)", "F6 Security-measures (P)", "F12 Outsider-disclosures-restricted (P)", "F1 Disclosure-in-negotiations (D)"],
  "TSC1": ["F4 Agreed-not-to-disclose (P)", "F6 Security-measures (P)", "F7 Brought-tools (P)"],
  "TSC2": ["F1 Disclosure-in-negotiations (D)", "F5 Agreement-not-specific (D)"]
}
Ensure each factor appears only once per list, even if mentioned multiple times in the argument.
"""

# --- LLM Interaction ---
def get_llm_response(client_type, model_name, system_prompt, user_prompt, max_tokens=DEFAULT_MAX_TOKENS):
    client = OPENAI_CLIENT if client_type == "openai" else GROQ_CLIENT

    current_max_tokens = max_tokens # Initialize with the passed-in max_tokens (or its default)
    extra_kwargs = {}               # Initialize as empty

    # Determine if JSON output is expected to correctly set reasoning_format
    is_json_mode = "JSON" in system_prompt.upper() or "JSON" in user_prompt.upper()

    # Special handling for specific Groq models
    # N.B. This model list might need updating if Groq adds more models requiring special handling.
    # The comment "# Added the maverick model here based on previous log errors" was on the original combined 'if',
    # it's retained here for historical context, though models are now handled more granularly.
    if client_type == "groq":
        if model_name == "qwen-qwq-32b":
            current_max_tokens = 5000
            # For qwen-qwq-32b, do not set reasoning_format as it previously caused "unexpected keyword argument" errors.
            # extra_kwargs remains empty for this specific parameter.
        elif model_name in ["deepseek-r1-distill-llama-70b", "llama-4-maverick-17b-128e-instruct"]:
            current_max_tokens = 5000
            if is_json_mode:
                # For JSON mode with these models, use "parsed".
                # "hidden" was in the original failing line for qwen; "parsed" is the other option for JSON.
                extra_kwargs["reasoning_format"] = "parsed"
            else:
                # For non-JSON mode, use "raw" based on user-provided successful example with deepseek.
                extra_kwargs["reasoning_format"] = "raw"
        # Other Groq models not in the specific lists above will use the initialized
        # current_max_tokens (i.e., default_max_tokens or passed max_tokens) and empty extra_kwargs.
    # For OpenAI client_type, current_max_tokens and extra_kwargs also use their initialized defaults from above.
    # The original 'else' branch that explicitly set these defaults for non-special-Groq cases
    # is now covered by the upfront initializations of current_max_tokens and extra_kwargs.

    common_params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": current_max_tokens, # Use potentially adjusted max_tokens
        "top_p": DEFAULT_TOP_P,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        **extra_kwargs # Add extra kwargs if any
    }
    try:
        response = client.chat.completions.create(**common_params)
        content = response.choices[0].message.content

        # Attempt to parse JSON if the prompt expects it
        if "JSON" in system_prompt.upper() or "JSON" in user_prompt.upper():
            try:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    start_index = content.find('{')
                    end_index = content.rfind('}')
                    if start_index != -1 and end_index != -1 and end_index > start_index:
                        json_str = content[start_index:end_index+1]
                    else:
                        print(f"Warning: Expected JSON but couldn't parse reliably from: {content}")
                        return content 
                
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} in content: {content}")
                return {"error": "Failed to parse JSON response", "raw_content": content}
        return content
    except Exception as e:
        print(f"Error calling LLM {model_name}: {e}")
        return {"error": str(e), "raw_content": ""}


# --- Scenario Parsing ---
def parse_scenario_text(scenario_text):
    """Parses a scenario string into a structured dictionary."""
    scenario = {
        "input_case_factors": [], "tsc1_factors": [], "tsc1_outcome": None,
        "tsc2_factors": [], "tsc2_outcome": None
    }
    current_section = None
    lines = scenario_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.lower().startswith("input scenario"):
            current_section = "input_case_factors"
        elif line.lower().startswith("tsc 1"):
            current_section = "tsc1_factors"
        elif line.lower().startswith("tsc 2"):
            current_section = "tsc2_factors"
        elif line.lower().startswith("outcome") and current_section == "tsc1_factors":
            scenario["tsc1_outcome"] = line.split()[-1].strip()
        elif line.lower().startswith("outcome") and current_section == "tsc2_factors":
            scenario["tsc2_outcome"] = line.split()[-1].strip()
        elif re.match(r"F\d+", line) and current_section:
            factor = line.split(',')[0].strip()
            if factor:
                 scenario[current_section].append(factor)
    return scenario

def format_scenario_for_prompt(parsed_scenario, for_plaintiff_tsc_key="tsc1", for_defendant_tsc_key="tsc2"):
    """Formats a parsed scenario dictionary back into a string for LLM prompts."""
    prompt_str = "Input Case:\n" + "\n".join(parsed_scenario["input_case_factors"]) + "\n\n"
    
    tsc1_data = parsed_scenario[for_plaintiff_tsc_key + "_factors"]
    tsc1_outcome = parsed_scenario[for_plaintiff_tsc_key + "_outcome"]
    prompt_str += f"TSC1 (for Plaintiff to cite):\noutcome {tsc1_outcome}\n" + "\n".join(tsc1_data) + "\n\n"
    
    tsc2_data = parsed_scenario[for_defendant_tsc_key + "_factors"]
    tsc2_outcome = parsed_scenario[for_defendant_tsc_key + "_outcome"]
    prompt_str += f"TSC2 (for Defendant to cite):\noutcome {tsc2_outcome}\n" + "\n".join(tsc2_data)
    return prompt_str

# --- Agent Architectures ---
def run_single_agent(client_type, model_name, parsed_scenario, enhanced_prompt_injection=None):
    system_prompt = BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT
    if enhanced_prompt_injection:
        system_prompt += "\n\nADDITIONAL INSTRUCTIONS:\n" + enhanced_prompt_injection
    
    user_prompt = format_scenario_for_prompt(parsed_scenario)
    
    argument = get_llm_response(client_type, model_name, system_prompt, user_prompt)
    return argument

def run_prompt_enhanced_single_agent(client_type, model_name, parsed_scenario):
    injection = COMBINED_ENHANCED_PROMPT_INJECTION
    return run_single_agent(client_type, model_name, parsed_scenario, enhanced_prompt_injection=injection)

def run_multi_agent_debate(client_type, model_name, parsed_scenario):
    full_argument = {}
    history = []
    
    # Plaintiff's Argument
    plaintiff_prompt_text = format_scenario_for_prompt(parsed_scenario) + \
                            "\n\nGenerate only the Plaintiff's Argument based on Input Case and TSC1."
    plaintiff_system_prompt = BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT.replace(
        "Construct a 3-Ply Argument:", 
        "You are the Plaintiff's advocate. Construct ONLY the Plaintiff's Argument using TSC1."
    ).replace("Output the 3-ply argument in a single JSON object...", "Output your argument part as a simple string.")
    plaintiff_arg_text = get_llm_response(client_type, model_name, plaintiff_system_prompt, plaintiff_prompt_text)
    if isinstance(plaintiff_arg_text, dict) and "error" in plaintiff_arg_text: plaintiff_arg_text = plaintiff_arg_text.get("raw_content", str(plaintiff_arg_text))
            
    full_argument["Plaintiff's Argument"] = plaintiff_arg_text
    history.append({"role": "Plaintiff", "content": plaintiff_arg_text})

    # Defendant's Counterargument
    defendant_prompt_text = format_scenario_for_prompt(parsed_scenario) + \
                           f"\n\nThe Plaintiff has argued:\n{plaintiff_arg_text}\n\nGenerate only the Defendant's Counterargument, distinguishing TSC1 and using TSC2."
    defendant_system_prompt = BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT.replace(
        "Construct a 3-Ply Argument:", 
        "You are the Defendant's advocate. Construct ONLY the Defendant's Counterargument using TSC2 and responding to Plaintiff."
    ).replace("Output the 3-ply argument in a single JSON object...", "Output your argument part as a simple string.")
    defendant_arg_text = get_llm_response(client_type, model_name, defendant_system_prompt, defendant_prompt_text)
    if isinstance(defendant_arg_text, dict) and "error" in defendant_arg_text: defendant_arg_text = defendant_arg_text.get("raw_content", str(defendant_arg_text))

    full_argument["Defendant's Counterargument"] = defendant_arg_text
    history.append({"role": "Defendant", "content": defendant_arg_text})

    # Plaintiff's Rebuttal
    rebuttal_prompt_text = format_scenario_for_prompt(parsed_scenario) + \
                          f"\n\nPrior arguments:\nPlaintiff: {plaintiff_arg_text}\nDefendant: {defendant_arg_text}\n\nGenerate only the Plaintiff's Rebuttal, addressing TSC2 and reinforcing the original argument."
    rebuttal_system_prompt = BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT.replace(
        "Construct a 3-Ply Argument:", 
        "You are the Plaintiff's advocate. Construct ONLY the Plaintiff's Rebuttal, responding to Defendant."
    ).replace("Output the 3-ply argument in a single JSON object...", "Output your argument part as a simple string.")
    rebuttal_arg_text = get_llm_response(client_type, model_name, rebuttal_system_prompt, rebuttal_prompt_text)
    if isinstance(rebuttal_arg_text, dict) and "error" in rebuttal_arg_text: rebuttal_arg_text = rebuttal_arg_text.get("raw_content", str(rebuttal_arg_text))

    full_argument["Plaintiff's Rebuttal"] = rebuttal_arg_text
    
    return full_argument

def _run_argument_ply_with_reflection(client_type, model_name, parsed_scenario, argument_type_key, tsc_to_cite_key, current_history_str, reflection_client_type, reflection_model_name):
    is_plaintiff_turn = "Plaintiff" in argument_type_key
    relevant_tsc_factors = parsed_scenario[tsc_to_cite_key + "_factors"]
    relevant_tsc_outcome = parsed_scenario[tsc_to_cite_key + "_outcome"]

    dev_system_prompt = BASE_ARGUMENT_DEVELOPER_SYSTEM_PROMPT.replace(
        "Construct a 3-Ply Argument:",
        f"You are an Argument Developer. Construct ONLY the {argument_type_key} using {tsc_to_cite_key.upper()}."
    ).replace("Output the 3-ply argument in a single JSON object...", f"Output your argument part for {argument_type_key} as a simple string.")
    
    dev_user_prompt = format_scenario_for_prompt(parsed_scenario) + \
                      f"\n\n{current_history_str}\n\nGenerate the {argument_type_key}. Ensure your output is just the text for this argument part."
    
    current_arg_text = get_llm_response(client_type, model_name, dev_system_prompt, dev_user_prompt)
    if isinstance(current_arg_text, dict) and "error" in current_arg_text: current_arg_text = current_arg_text.get("raw_content", str(current_arg_text))

    for i in range(REFLECTION_THRESHOLD + 1):
        if isinstance(current_arg_text, str) and "TERMINATE:" in current_arg_text.upper():
            break

        analyst_user_prompt = f"""Argument Segment to Analyze ({argument_type_key}):
{current_arg_text}

Input Case Factors:
{json.dumps(parsed_scenario["input_case_factors"], indent=2)}

Cited TSC ({tsc_to_cite_key.upper()}) Factors:
{json.dumps(relevant_tsc_factors, indent=2)}
Actual Outcome of {tsc_to_cite_key.upper()}: {relevant_tsc_outcome}
"""
        factor_analysis = get_llm_response(reflection_client_type, reflection_model_name, FACTOR_ANALYST_SYSTEM_PROMPT, analyst_user_prompt)

        if isinstance(factor_analysis, dict) and factor_analysis.get("analysis_outcome") == "REQUIRES_ABSTENTION":
            abstention_reason = factor_analysis.get('summary', "Factor Analyst required abstention.")
            print(f"INFO ({argument_type_key} - Reflection {i+1}): Factor Analyst requires abstention. Stopping reflection for this ply. Reason: {abstention_reason}")
            current_arg_text = f"TERMINATE: Generation stopped. {abstention_reason}"
            break

        polisher_user_prompt = f"""Argument Segment ({argument_type_key}):
{current_arg_text}

Input Case Factors: {json.dumps(parsed_scenario["input_case_factors"])}
TSC1 Factors: {json.dumps(parsed_scenario["tsc1_factors"])} (Outcome: {parsed_scenario["tsc1_outcome"]})
TSC2 Factors: {json.dumps(parsed_scenario["tsc2_factors"])} (Outcome: {parsed_scenario["tsc2_outcome"]})

Factor Analyst's Report:
{json.dumps(factor_analysis, indent=2)}
"""
        polisher_feedback = get_llm_response(reflection_client_type, reflection_model_name, ARGUMENT_POLISHER_SYSTEM_PROMPT, polisher_user_prompt)
        
        needs_revision = False
        revision_instructions = ""
        if isinstance(polisher_feedback, dict):
            needs_revision = polisher_feedback.get("revision_needed", False)
            revision_instructions = polisher_feedback.get("instructions_for_developer", "")
        else: # Polisher might return non-JSON error string
            print(f"Warning: Polisher feedback was not a dict: {polisher_feedback}")

        if needs_revision and i < REFLECTION_THRESHOLD:
            revise_system_prompt = dev_system_prompt + "\n\nYou received feedback. Revise your previous argument for {argument_type_key}. Output only the revised text for this part."
            revise_user_prompt = f"""Original Scenario & History:
{format_scenario_for_prompt(parsed_scenario)}
{current_history_str}

Your previous attempt for {argument_type_key}:
{current_arg_text}

Feedback from Argument Polisher:
{revision_instructions}

Factor Analyst's Report for context:
{json.dumps(factor_analysis)}

Generate the revised {argument_type_key}. Output only the text for this argument part."""
            
            current_arg_text = get_llm_response(client_type, model_name, revise_system_prompt, revise_user_prompt)
            if isinstance(current_arg_text, dict) and "error" in current_arg_text: current_arg_text = current_arg_text.get("raw_content", str(current_arg_text))
            print(f"INFO ({argument_type_key} - Reflection {i+1}): Attempted revision based on feedback.")
        else:
            if not needs_revision:
                 print(f"INFO ({argument_type_key} - Reflection {i+1}): Polisher deemed no further revision needed.")
            elif i >= REFLECTION_THRESHOLD:
                 print(f"INFO ({argument_type_key}): Max reflection threshold ({REFLECTION_THRESHOLD}) reached.")
            break
            
    return current_arg_text


def run_multi_agent_debate_with_reflection(client_type, model_name, parsed_scenario, reflection_client_type, reflection_model_name):
    full_argument = {}
    history_str = "Argument history so far:\n"

    plaintiff_arg = _run_argument_ply_with_reflection(
        client_type, model_name, parsed_scenario, "Plaintiff's Argument", "tsc1", 
        history_str, reflection_client_type, reflection_model_name
    )
    full_argument["Plaintiff's Argument"] = plaintiff_arg
    history_str += f"- Plaintiff's Argument: {plaintiff_arg}\n"

    defendant_arg = _run_argument_ply_with_reflection(
        client_type, model_name, parsed_scenario, "Defendant's Counterargument", "tsc2",
        history_str, reflection_client_type, reflection_model_name
    )
    full_argument["Defendant's Counterargument"] = defendant_arg
    history_str += f"- Defendant's Counterargument: {defendant_arg}\n"

    rebuttal_arg = _run_argument_ply_with_reflection(
        client_type, model_name, parsed_scenario, "Plaintiff's Rebuttal", "tsc1",
        history_str, reflection_client_type, reflection_model_name
    )
    full_argument["Plaintiff's Rebuttal"] = rebuttal_arg
    
    return full_argument

# --- Evaluation ---
def distill_factors(distiller_client_type, distiller_model_name, argument_obj):
    """Uses an LLM to distill factors from a generated argument."""
    argument_json_str = json.dumps(argument_obj) if isinstance(argument_obj, dict) else str(argument_obj) # Ensure it's a string for the prompt

    distilled = get_llm_response(distiller_client_type, distiller_model_name, FACTOR_DISTILLER_SYSTEM_PROMPT, argument_json_str, max_tokens=1000)
    
    # Ensure distilled is a dictionary and normalize factor lists
    if isinstance(distilled, dict) and "error" not in distilled:
        for key in ["Input Case", "TSC1", "TSC2"]:
            if key in distilled and isinstance(distilled[key], list):
                distilled[key] = sorted(list(set(str(f).strip() for f in distilled[key] if isinstance(f, str) and f.strip())))
            else:
                distilled[key] = []
        return distilled
        
    print(f"Factor distillation failed or returned error/unexpected type: {type(distilled)} {distilled}")
    return {"Input Case": [], "TSC1": [], "TSC2": []}

def calculate_precedent_identification_accuracy(distilled_factors, ground_truth_scenario):
    mismatches = 0
    total_factors_considered = 0

    gt_input = set(ground_truth_scenario["input_case_factors"])
    gt_tsc1_factors = set(ground_truth_scenario["tsc1_factors"])
    gt_tsc1_outcome = ground_truth_scenario["tsc1_outcome"].lower()
    gt_tsc2_factors = set(ground_truth_scenario["tsc2_factors"])
    gt_tsc2_outcome = ground_truth_scenario["tsc2_outcome"].lower()

    # Check Plaintiff's use of TSC1
    # Common factors claimed between Input and TSC1 in the distilled output
    distilled_input_for_tsc1 = set(distilled_factors.get("Input Case", []))
    distilled_tsc1 = set(distilled_factors.get("TSC1", []))
    
    # Factors from distilled_input that are also in distilled_tsc1
    claimed_common_input_tsc1 = distilled_input_for_tsc1.intersection(distilled_tsc1)
    
    for factor in claimed_common_input_tsc1:
        total_factors_considered += 1
        if not (factor in gt_input and factor in gt_tsc1_factors):
            mismatches += 1
            
    # Outcome check for TSC1 (should be plaintiff)
    if claimed_common_input_tsc1: # Only check outcome if precedent was actually cited/used
        total_factors_considered +=1
        if gt_tsc1_outcome != "plaintiff":
            mismatches +=1

    # Check Defendant's use of TSC2
    distilled_input_for_tsc2 = set(distilled_factors.get("Input Case", []))
    distilled_tsc2 = set(distilled_factors.get("TSC2", []))
    claimed_common_input_tsc2 = distilled_input_for_tsc2.intersection(distilled_tsc2)

    for factor in claimed_common_input_tsc2:
        total_factors_considered +=1
        if not (factor in gt_input and factor in gt_tsc2_factors):
            mismatches += 1
            
    if claimed_common_input_tsc2:
        total_factors_considered +=1
        if gt_tsc2_outcome != "defendant":
            mismatches +=1
            
    if total_factors_considered == 0: return 0.0
    accuracy_pi = (1 - (mismatches / total_factors_considered)) * 100
    return max(0, accuracy_pi)

def calculate_hallucination_accuracy_and_factor_recall(distilled_factors, ground_truth_scenario):
    gt_input = set(ground_truth_scenario["input_case_factors"])
    gt_tsc1 = set(ground_truth_scenario["tsc1_factors"])
    gt_tsc2 = set(ground_truth_scenario["tsc2_factors"])
    
    claimed_input = set(distilled_factors.get("Input Case", []))
    claimed_tsc1 = set(distilled_factors.get("TSC1", []))
    claimed_tsc2 = set(distilled_factors.get("TSC2", []))

    # --- Hallucination Accuracy (REVISED) ---
    # Calculate based on the proportion of *claimed* factors that were correct for their specific case.
    
    hallucinated_in_input = len(claimed_input - gt_input)
    hallucinated_in_tsc1 = len(claimed_tsc1 - gt_tsc1)
    hallucinated_in_tsc2 = len(claimed_tsc2 - gt_tsc2)
    
    total_hallucinated_factors = hallucinated_in_input + hallucinated_in_tsc1 + hallucinated_in_tsc2
    total_claimed_factors_count = len(claimed_input) + len(claimed_tsc1) + len(claimed_tsc2)

    if total_claimed_factors_count == 0:
        # If nothing was claimed, there were no hallucinations.
        hallucination_accuracy = 100.0
    else:
        hallucination_accuracy = (1 - (total_hallucinated_factors / total_claimed_factors_count)) * 100

    # --- Factor Utilization Recall (REVISED PREVIOUSLY) ---
    # Calculate recall based on factors recalled *within their specific case context*.
    
    total_gt_factors_count_per_case = len(gt_input) + len(gt_tsc1) + len(gt_tsc2)
    
    if total_gt_factors_count_per_case == 0:
        # If ground truth has no factors, recall is vacuously 100%
        factor_utilization_recall = 100.0
    else:
        missing_in_input = len(gt_input - claimed_input)
        missing_in_tsc1 = len(gt_tsc1 - claimed_tsc1)
        missing_in_tsc2 = len(gt_tsc2 - claimed_tsc2)
        
        total_missing_factors_per_case = missing_in_input + missing_in_tsc1 + missing_in_tsc2
        
        factor_utilization_recall = (1 - (total_missing_factors_per_case / total_gt_factors_count_per_case)) * 100
    
    return {
        "hallucination_accuracy": max(0, hallucination_accuracy),
        "factor_utilization_recall": max(0, factor_utilization_recall)
    }

def calculate_successful_abstention(distilled_factors):
    """
    Determines successful abstention if distilled_factors equals
    {"Input Case": [], "TSC1": [], "TSC2": []}.
    """
    expected_empty_structure = {"Input Case": [], "TSC1": [], "TSC2": []}
    if distilled_factors == expected_empty_structure:
        return 1
    return 0

# --- Logging ---
LOG_FILE_HANDLE = None
def setup_logging(log_filename_prefix="experiment_log", model_name="unknown_model"):
    global LOG_FILE_HANDLE
    # Sanitize model name for filename
    safe_model_name = re.sub(r'[^\w\-_\.]', '_', model_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_filename_prefix}_{safe_model_name}_{timestamp}.jsonl"
    LOG_FILE_HANDLE = open(log_file, 'a')
    print(f"Logging to {log_file}")

def log_experiment_data(data):
    if LOG_FILE_HANDLE:
        LOG_FILE_HANDLE.write(json.dumps(data) + "\n")
        LOG_FILE_HANDLE.flush()

def close_logging():
    if LOG_FILE_HANDLE:
        LOG_FILE_HANDLE.close()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run legal argumentation experiments.")
    # Removed model_type argument - will be derived from model_name
    parser.add_argument("--model_name", default="gpt-4o-mini", help="Name of the LLM to use for arguments.")
    parser.add_argument("--reflection_model_name", default="gpt-4.1", help="Name of LLM for reflection agents (Factor Analyst, Polisher).")
    # Removed reflection_client_type argument - will be derived
    parser.add_argument("--distiller_model_name", default="gpt-4.1", help="Name of LLM for Factor Distiller agent.")
    # Removed distiller_client_type argument - will be derived
    parser.add_argument("--num_scenarios_per_type", type=int, default=1, help="Number of scenarios to generate per type (non-arguable, mismatched, arguable). Min 1.")
    parser.add_argument("--complexity", type=int, default=3, help="Complexity for scenario generation (number of factors). Min 1.")
    parser.add_argument("--output_format", choices=['factor', 'code'], default='factor')
    parser.add_argument("--log_prefix", default="multi_agent_factor_experiment")

    args = parser.parse_args()

    # --- Auto-determine client types ---
    KNOWN_OPENAI_MODELS = {
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1", "gpt-3.5-turbo"
        # Add any other OpenAI model identifiers used
    }

    def determine_client_type(model_name):
        # Check if the model name starts with any known OpenAI prefix or is an exact match
        if any(model_name.startswith(prefix) for prefix in KNOWN_OPENAI_MODELS) or model_name in KNOWN_OPENAI_MODELS:
            return "openai"
        else:
            # Assume Groq if not OpenAI (adjust if more client types are added)
            return "groq"

    model_client_type = determine_client_type(args.model_name)
    reflection_client_type = determine_client_type(args.reflection_model_name)
    distiller_client_type = determine_client_type(args.distiller_model_name)

    print(f"Determined client types: Main={model_client_type}, Reflection={reflection_client_type}, Distiller={distiller_client_type}")
    # --- End auto-determination ---


    setup_logging(args.log_prefix, args.model_name)
    # reflection_client_type = args.reflection_client_type # No longer needed, derived above

    scenario_types_map = {
        "non-arguable": "non-arguable",
        "mismatched": "mismatched",
        "arguable": "arguable"
    }

    for scenario_key, scenario_mode_for_gen in scenario_types_map.items():
        print(f"\n--- Generating {args.num_scenarios_per_type} '{scenario_key}' scenarios (complexity {args.complexity}) ---")
        for i in range(args.num_scenarios_per_type):
            print(f"\nProcessing {scenario_key} scenario {i+1}/{args.num_scenarios_per_type}")
            gen = ScenarioGenerator(mode1=scenario_mode_for_gen, mode2=scenario_mode_for_gen, 
                                    output_format=args.output_format, complexity=args.complexity)
            scenario_text = gen.generate_initial_prompt()
            parsed_scenario_gt = parse_scenario_text(scenario_text)

            experiment_data_base = {
                "scenario_type": scenario_key,
                "scenario_id": i + 1,
                "complexity": args.complexity,
                "ground_truth_scenario_text": scenario_text,
                "ground_truth_parsed": parsed_scenario_gt,
                "model_type": model_client_type, # Use derived type
                "model_name": args.model_name,
                "reflection_model_name": args.reflection_model_name,
                "distiller_model_name": args.distiller_model_name,
                "distiller_client_type": distiller_client_type, # Use derived type
            }

            architectures_to_run = {
                "single_agent": {
                    "type": "single_run",
                    "func": lambda: run_single_agent(model_client_type, args.model_name, parsed_scenario_gt) # Pass derived type
                },
                "prompt_enhanced_single_agent": {
                    "type": "single_run",
                    "func": lambda: run_prompt_enhanced_single_agent(model_client_type, args.model_name, parsed_scenario_gt) # Pass derived type
                },
                "multi_agent_debate": {
                    "type": "single_run",
                    "func": lambda: run_multi_agent_debate(model_client_type, args.model_name, parsed_scenario_gt) # Pass derived type
                },
                "multi_agent_debate_with_reflection": {
                    "type": "single_run",
                    "func": lambda: run_multi_agent_debate_with_reflection(model_client_type, args.model_name, parsed_scenario_gt, reflection_client_type, args.reflection_model_name) # Pass derived types
                },
            }

            for arch_name_key, arch_config in architectures_to_run.items():
                runs_for_this_arch = []
                if arch_config["type"] == "multi_task":
                    for task_type in arch_config["task_types"]:
                        runs_for_this_arch.append({
                            "name": f"{arch_name_key}_{task_type}",
                            "func": lambda tt=task_type: arch_config["func_template"](tt) # Capture task_type via default argument
                        })
                else: # single_run
                    runs_for_this_arch.append({
                        "name": arch_name_key,
                        "func": arch_config["func"]
                    })

                for run_info in runs_for_this_arch:
                    arch_name_to_log_and_print = run_info["name"]
                    arch_executable_func = run_info["func"]

                    print(f"  Running architecture: {arch_name_to_log_and_print}...")
                    current_run_data = {**experiment_data_base, "architecture": arch_name_to_log_and_print}
                    start_time = time.time()
                    generated_argument = arch_executable_func()
                    current_run_data["generation_time_seconds"] = time.time() - start_time
                    current_run_data["generated_argument_json_or_text"] = generated_argument

                    # Distill factors
                    distilled_arg_factors = distill_factors(distiller_client_type, args.distiller_model_name, generated_argument) # Pass derived type
                    current_run_data["distilled_factors"] = distilled_arg_factors
                    
                    # Calculate metrics
                    pi_acc = calculate_precedent_identification_accuracy(distilled_arg_factors, parsed_scenario_gt)
                    arg_metrics = calculate_hallucination_accuracy_and_factor_recall(distilled_arg_factors, parsed_scenario_gt)
                    
                    current_run_data["precedent_identification_accuracy"] = pi_acc
                    current_run_data["hallucination_accuracy"] = arg_metrics["hallucination_accuracy"]
                    current_run_data["factor_utilization_recall"] = arg_metrics["factor_utilization_recall"]
                    
                    log_message_parts = []
                    # Group non-arguable and mismatched scenarios for abstention check
                    if scenario_key == "non-arguable" or scenario_key == "mismatched":
                        successful_abstention_val = calculate_successful_abstention(distilled_arg_factors)
                        current_run_data["successful_abstention"] = successful_abstention_val
                        log_message_parts.append(f"Successful Abstention: {successful_abstention_val*100:.2f}%")
                        
                        # If abstention was successful, override hallucination/recall
                        if successful_abstention_val == 1:
                            current_run_data["hallucination_accuracy"] = 100.0
                            current_run_data["factor_utilization_recall"] = 0.0
                            # Update log message parts for clarity
                            log_message_parts.append("Halluc Acc: 100.00% (Forced by Abstention)")
                            log_message_parts.append("Factor Recall: 0.00% (Forced by Abstention)")
                            # PI Acc is not logged when abstention occurs
                        else:
                             # Log calculated values if abstention failed for non-arguable/mismatched
                            # Include PI Acc here as it was previously logged for mismatched when no abstention happened
                            log_message_parts.append(f"PI Acc: {pi_acc:.2f}%") 
                            log_message_parts.append(f"Halluc Acc: {current_run_data['hallucination_accuracy']:.2f}%")
                            log_message_parts.append(f"Factor Recall: {current_run_data['factor_utilization_recall']:.2f}%")
                    else: # Handles only "arguable" scenarios now
                        # For arguable, log calculated values normally, successful_abstention is not the primary metric
                        current_run_data["successful_abstention"] = 0 # Mark as not applicable/expected for this type
                        log_message_parts.append(f"PI Acc: {pi_acc:.2f}%")
                        log_message_parts.append(f"Halluc Acc: {arg_metrics['hallucination_accuracy']:.2f}%")
                        log_message_parts.append(f"Factor Recall: {arg_metrics['factor_utilization_recall']:.2f}%")
                    
                    log_experiment_data(current_run_data)
                    print(f"    Done. {', '.join(log_message_parts)}")
                    time.sleep(random.uniform(1,3)) # Rate limiting

    close_logging()
    print("\nExperiments finished.")

if __name__ == "__main__":
    main()
