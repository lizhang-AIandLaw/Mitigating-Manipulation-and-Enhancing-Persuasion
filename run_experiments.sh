#!/bin/bash

# Define the models to run
# You can customize this list with the models you want to test.
# Ensure your .env file has the necessary API keys for these models.
MODELS=(
    "llama3-8b-8192"
    "llama3-70b-8192"
)

# Default values for other arguments (can be overridden here or passed as script arguments)
# These are based on the defaults in your Python script.
REFLECTION_MODEL="gpt-4.1"
DISTILLER_MODEL="gpt-4.1"
NUM_SCENARIOS=90
COMPLEXITY=5
OUTPUT_FORMAT="factor"
LOG_PREFIX_BASE="multi_agent_experiment" # Base for log file names

# Path to the python script
PYTHON_SCRIPT="multi_agent_factor.py"

# Check if python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found at the expected location."
    echo "Current directory: $(pwd)"
    echo "Please ensure the path is correct or run this script from the workspace root."
    exit 1
fi

# Loop through the models and run the experiment
for model_name in "${MODELS[@]}"; do
    echo "----------------------------------------------------"
    echo "Starting experiment for model: $model_name"
    echo "----------------------------------------------------"

    # Sanitize model name for use in log prefix (replaces non-alphanumeric with underscore)
    sane_model_name=$(echo "$model_name" | sed 's/[^a-zA-Z0-9_]/_/g')
    current_log_prefix="${LOG_PREFIX_BASE}_${sane_model_name}"

    # Construct and execute the command
    # Ensure your Python script has execute permissions or call it with 'python' or 'python3'
    python3 "$PYTHON_SCRIPT" \
        --model_name "$model_name" \
        --reflection_model_name "$REFLECTION_MODEL" \
        --distiller_model_name "$DISTILLER_MODEL" \
        --num_scenarios_per_type "$NUM_SCENARIOS" \
        --complexity "$COMPLEXITY" \
        --output_format "$OUTPUT_FORMAT" \
        --log_prefix "$current_log_prefix"

    # Check the exit status of the python script
    if [ $? -ne 0 ]; then
        echo "----------------------------------------------------"
        echo "ERROR: Experiment for model '$model_name' failed."
        echo "----------------------------------------------------"
        # Decide if you want to stop the script on error or continue with the next model
        # exit 1 # Uncomment to stop the entire script on the first error
    else
        echo "----------------------------------------------------"
        echo "Finished experiment for model: $model_name"
        echo "Logs should be prefixed with: $current_log_prefix"
        echo "----------------------------------------------------"
    fi
    
    # Optional: Add a small delay between runs to avoid rate limiting or to allow system to settle
    # sleep 5 
done

echo "All experiments finished."
echo "Check the log files prefixed with '${LOG_PREFIX_BASE}_[model_name]' for results." 