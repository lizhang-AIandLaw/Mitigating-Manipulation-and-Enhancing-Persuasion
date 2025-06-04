import random
import re
import json
import csv
import string
import argparse
import sys

class ScenarioGenerator():
    # Map new mode names to internal mode names
    MODE_MAPPING = {
        "non-arguable": "unarguable",
        "mismatched": "mismatched",
        "arguable": "arguable"
    }
    
    def __init__(self, mode1="mismatched", mode2="mismatched", output_format="factor", complexity=5):
        print(f"Initializing generator with mode1={mode1}, mode2={mode2}, format={output_format}, complexity={complexity}")
        self.factors = [
            "F1 Disclosure-in-negotiations (D)", 
            "F2 Bribe-employee (P)", 
            "F3 Employee-sole-developer (D)", 
            "F4 Agreed-not-to-disclose (P)", 
            "F5 Agreement-not-specific (D)", 
            "F6 Security-measures (P)", 
            "F7 Brought-tools (P)", 
            "F8 Competitive-advantage (P)", 
            "F10 Secrets-disclosed-outsiders (D)", 
            "F11 Vertical-knowledge (D)", 
            "F12 Outsider-disclosures-restricted (P)", 
            "F13 Noncompetition-agreement (P)", 
            "F14 Restricted-materials-used (P)", 
            "F15 Unique-product (P)", 
            "F16 Info-reverse-engineerable (D)", 
            "F17 Info-independently-generated (D)", 
            "F18 Identical-products (P)", 
            "F19 No-security-measures (D)", 
            "F20 Info-known-to-competitors (D)", 
            "F21 Knew-info-confidential (P)", 
            "F22 Invasive-techniques (P)", 
            "F23 Waiver-of-confidentiality (D)", 
            "F24 Info-obtainable-elsewhere (D)", 
            "F25 Info-reverse-engineered (D)", 
            "F26 Deception (P)", 
            "F27 Disclosure-in-public-forum (D)"
        ]
        
        # Handle direct mode strings or do mapping if needed
        if mode1 in self.MODE_MAPPING:
            self.mode1 = self.MODE_MAPPING[mode1]
        elif mode1 in self.MODE_MAPPING.values():
            self.mode1 = mode1
        else:
            self.mode1 = "mismatched"  # Default
            
        if mode2 in self.MODE_MAPPING:
            self.mode2 = self.MODE_MAPPING[mode2]
        elif mode2 in self.MODE_MAPPING.values():
            self.mode2 = mode2
        else:
            self.mode2 = "mismatched"  # Default
            
        print(f"Internal modes: mode1={self.mode1}, mode2={self.mode2}")
        self.output_format = output_format
        self.complexity = complexity
        print("Generating initial scenario...")
        self.input_factors, self.tsc1, self.tsc2 = self.generate_input_scenario()
        print("Initialization complete.")

    def generate_input_factor(self):
        print("Generating input factors...")
        selected_factors = []
        used_indices = set()
        
        min_factors = max(1, self.complexity - 1)
        max_factors = self.complexity + 1
        target_count = random.randint(min_factors, max_factors)
        print(f"Target count for input factors: {target_count}")
        
        while len(selected_factors) < target_count:
            index = random.randint(0, len(self.factors) - 1)
            if index not in used_indices:
                selected_factors.append(self.factors[index])
                used_indices.add(index)
        
        print(f"Generated {len(selected_factors)} input factors")
        return selected_factors

    def generate_tsc_factor(self, input_factors, mode="unarguable", is_tsc1=True):
        tsc_type = "TSC1" if is_tsc1 else "TSC2"
        print(f"Generating {tsc_type} factors with mode: {mode}...")
        selected_factors = []
        used_indices = set()
        
        min_factors = max(1, self.complexity - 1)
        max_factors = self.complexity + 1
        target_count = random.randint(min_factors, max_factors)
        print(f"Target count for {tsc_type}: {target_count}")
        
        if mode == "arguable":
            # Get pro-defendant and pro-plaintiff factors from input
            input_d_factors = [f for f in input_factors if "(D)" in f]
            input_p_factors = [f for f in input_factors if "(P)" in f]
            
            if is_tsc1:
                # For TSC1: 1-3 pro-plaintiff factors
                if input_p_factors:  # Check if there are any P factors
                    n_common = random.randint(1, min(3, len(input_p_factors)))
                    common_factors = random.sample(input_p_factors, n_common)
                    selected_factors.extend(common_factors)
                    used_indices.update(self.factors.index(f) for f in common_factors)
            else:
                # For TSC2: 1-3 pro-defendant factors 
                if input_d_factors:  # Check if there are any D factors
                    n_common = random.randint(1, min(3, len(input_d_factors)))
                    common_factors = random.sample(input_d_factors, n_common)
                    selected_factors.extend(common_factors)
                    used_indices.update(self.factors.index(f) for f in common_factors)
            
            # Maybe add some other shared factors
            other_factors = [f for f in input_factors if f not in selected_factors]
            if other_factors and random.random() < 0.7: # Increased probability
                n_other = random.randint(1, min(3, len(other_factors))) # Increased max number
                selected_factors.extend(random.sample(other_factors, n_other))
            
            # Add random factors to reach desired length
            while len(selected_factors) < target_count:
                factor = random.choice(self.factors)
                if factor not in selected_factors:
                    selected_factors.append(factor)
        else:
            # Default unarguable mode: ensure absolutely no common factors
            print(f"Using unarguable mode for {tsc_type}")
            available_factors = [f for f in self.factors if f not in input_factors]
            print(f"Available unique factors: {len(available_factors)}")
            
            # Verify we have enough unique factors
            if len(available_factors) < target_count:
                # Adjust the target count if we don't have enough unique factors
                print(f"Not enough unique factors. Adjusting target count from {target_count} to {len(available_factors)}")
                target_count = len(available_factors)
            
            # Select random factors from available factors (none from input)
            try:
                selected_factors = random.sample(available_factors, target_count)
                print(f"Selected {len(selected_factors)} factors for {tsc_type}")
            except ValueError as e:
                print(f"Error selecting factors: {e}")
                print(f"Available factors: {len(available_factors)}, Target count: {target_count}")
                # Fallback selection
                selected_factors = available_factors[:target_count]
                    
        return selected_factors

    def find_common_factors(self, tsc):
        if tsc == "tsc1":
            tsc_factor = self.tsc1
        elif tsc == "tsc2":
            tsc_factor = self.tsc2
        common_factors = [factor for factor in self.input_factors if factor in tsc_factor]

        return common_factors

    def generate_input_scenario(self):
        print("Generating complete scenario...")
        input_factors = self.generate_input_factor()
        print(f"Input factors: {len(input_factors)}")
        
        # For mismatched mode, we'll generate both TSCs using arguable mode first, then swap
        if self.mode1 == "mismatched" and self.mode2 == "mismatched":
            print("Using mismatched mode: generating with arguable mode then swapping outcomes")
            print("Generating TSC1 with arguable mode")
            tsc1 = self.generate_tsc_factor(input_factors, mode="arguable", is_tsc1=True)
            
            print("Generating TSC2 with arguable mode")
            tsc2 = self.generate_tsc_factor(input_factors, mode="arguable", is_tsc1=False)
            
            # Swap TSC1 and TSC2 to implement mismatched mode
            print("Swapping TSC1 and TSC2 for mismatched mode")
            tsc1, tsc2 = tsc2, tsc1
        else:
            # Standard generation for other modes
            print(f"Generating TSC1 with mode: {self.mode1}")
            tsc1 = self.generate_tsc_factor(input_factors, mode=self.mode1, is_tsc1=True)
            
            print(f"Generating TSC2 with mode: {self.mode2}")
            tsc2 = self.generate_tsc_factor(input_factors, mode=self.mode2, is_tsc1=False)
        
        # For unarguable mode, double-check that there are no overlaps
        if self.mode1 == "unarguable":
            common_factors = [f for f in input_factors if f in tsc1]
            if common_factors:
                print(f"Found overlapping factors in TSC1: {len(common_factors)}. Regenerating.")
                tsc1 = self.generate_tsc_factor(input_factors, mode=self.mode1, is_tsc1=True)
            else:
                print("No overlaps found in TSC1.")
                
        if self.mode2 == "unarguable":
            common_factors = [f for f in input_factors if f in tsc2]
            if common_factors:
                print(f"Found overlapping factors in TSC2: {len(common_factors)}. Regenerating.")
                tsc2 = self.generate_tsc_factor(input_factors, mode=self.mode2, is_tsc1=False)
            else:
                print("No overlaps found in TSC2.")
        
        return input_factors, tsc1, tsc2

    def extract_factor_number(self, factor):
        match = re.match(r"F(\d+)", factor)
        return int(match.group(1)) if match else float('inf')

    def generate_initial_prompt(self):
        print("Generating prompt...")
        if self.output_format == "code":
            print("Warning: output_format is 'code' but FACTOR_CODES has been removed. Using factor names instead.")
            input_scenario_items = [f"{factor}" for factor in sorted(self.input_factors, key=lambda x: int(re.search(r'\d+', x).group()))]
            tsc1_factors_items = [f"{factor}" for factor in sorted(self.tsc1, key=lambda x: int(re.search(r'\d+', x).group()))]
            tsc2_factors_items = [f"{factor}" for factor in sorted(self.tsc2, key=lambda x: int(re.search(r'\d+', x).group()))]
        else:
            # Use original factor names
            input_scenario_items = sorted(self.input_factors, key=lambda x: int(re.search(r'\d+', x).group()))
            tsc1_factors_items = sorted(self.tsc1, key=lambda x: int(re.search(r'\d+', x).group()))
            tsc2_factors_items = sorted(self.tsc2, key=lambda x: int(re.search(r'\d+', x).group()))
        
        input_scenario_str = ",\n\t".join(input_scenario_items)
        tsc1_factors_str = ",\n\t".join(tsc1_factors_items)
        tsc2_factors_str = ",\n\t".join(tsc2_factors_items)

        # For mismatched mode, we swap the outcomes
        if self.mode1 == "mismatched" and self.mode2 == "mismatched":
            input_scenario_prompt = f"""
Input Scenario 
\t{input_scenario_str}

TSC 1
outcome Defendant
\t{tsc1_factors_str}

TSC 2
outcome Plaintiff
\t{tsc2_factors_str}
"""
        else:
            input_scenario_prompt = f"""
Input Scenario 
\t{input_scenario_str}

TSC 1
outcome Plaintiff
\t{tsc1_factors_str}

TSC 2
outcome Defendant
\t{tsc2_factors_str}
"""
        return input_scenario_prompt

    def update_tsc(self, tsc_name, mode="citable"):
        # Handle mode mapping if needed
        if mode in self.MODE_MAPPING:
            internal_mode = self.MODE_MAPPING[mode]
        elif mode in self.MODE_MAPPING.values():
            internal_mode = mode
        else:
            internal_mode = "mismatched"  # Default
            
        print(f"Updating {tsc_name} with mode {internal_mode}")
        
        if internal_mode == "mismatched":
            # For mismatched mode, generate both TSCs with arguable mode then swap
            print("Using mismatched mode: generating with arguable mode then swapping outcomes")
            self.tsc1 = self.generate_tsc_factor(self.input_factors, mode="arguable", is_tsc1=True)
            self.tsc2 = self.generate_tsc_factor(self.input_factors, mode="arguable", is_tsc1=False)
            # Swap TSC1 and TSC2 to implement mismatched mode
            print("Swapping TSC1 and TSC2 for mismatched mode")
            self.tsc1, self.tsc2 = self.tsc2, self.tsc1
        else:
            # Standard generation for other modes
            if tsc_name == "tsc1":
                self.tsc1 = self.generate_tsc_factor(self.input_factors, mode=internal_mode, is_tsc1=True)
                # For unarguable mode, ensure no overlaps
                if internal_mode == "unarguable":
                    attempts = 0
                    while any(factor in self.input_factors for factor in self.tsc1) and attempts < 5:
                        print(f"Found overlapping factors in TSC1 update. Regenerating (attempt {attempts+1}).")
                        self.tsc1 = self.generate_tsc_factor(self.input_factors, mode=internal_mode, is_tsc1=True)
                        attempts += 1
                    if attempts >= 5:
                        print("Warning: Could not eliminate overlaps after 5 attempts.")
            elif tsc_name == "tsc2":
                self.tsc2 = self.generate_tsc_factor(self.input_factors, mode=internal_mode, is_tsc1=False)
                # For unarguable mode, ensure no overlaps
                if internal_mode == "unarguable":
                    attempts = 0
                    while any(factor in self.input_factors for factor in self.tsc2) and attempts < 5:
                        print(f"Found overlapping factors in TSC2 update. Regenerating (attempt {attempts+1}).")
                        self.tsc2 = self.generate_tsc_factor(self.input_factors, mode=internal_mode, is_tsc1=False)
                        attempts += 1
                    if attempts >= 5:
                        print("Warning: Could not eliminate overlaps after 5 attempts.")
            else:
                raise ValueError("Invalid TSC name. Use 'tsc1' or 'tsc2'.")

        if self.output_format == "code":
            print("Warning: output_format is 'code' but FACTOR_CODES has been removed. Using factor names instead.")
            input_scenario_items = [f"{factor}" for factor in sorted(self.input_factors, key=self.extract_factor_number)]
            tsc1_factors_items = [f"{factor}" for factor in sorted(self.tsc1, key=self.extract_factor_number)]
            tsc2_factors_items = [f"{factor}" for factor in sorted(self.tsc2, key=self.extract_factor_number)]
        else:
            # Use original factor names
            input_scenario_items = sorted(self.input_factors, key=self.extract_factor_number)
            tsc1_factors_items = sorted(self.tsc1, key=self.extract_factor_number)
            tsc2_factors_items = sorted(self.tsc2, key=self.extract_factor_number)
        
        input_scenario_str = ",\n\t".join(input_scenario_items)
        tsc1_factors_str = ",\n\t".join(tsc1_factors_items)
        tsc2_factors_str = ",\n\t".join(tsc2_factors_items)

        # For mismatched mode, we swap the outcomes
        if internal_mode == "mismatched":
            input_scenario_prompt = f"""
Input Scenario 
\t{input_scenario_str}

TSC 1
outcome Defendant
\t{tsc1_factors_str}

TSC 2
outcome Plaintiff
\t{tsc2_factors_str}
"""
        else:
            input_scenario_prompt = f"""
Input Scenario 
\t{input_scenario_str}

TSC 1
outcome Plaintiff
\t{tsc1_factors_str}

TSC 2
outcome Defendant
\t{tsc2_factors_str}
"""
        return input_scenario_prompt

    @classmethod
    def get_factor_mapping(cls):
        # Return an empty dictionary or handle as appropriate since FACTOR_CODES is removed
        return {}

    def restart(self):
        print("Restarting scenario generation...")
        self.input_factors, self.tsc1, self.tsc2 = self.generate_input_scenario()
        print("Restart complete.")

def generate_datasets(mode="non-arguable", output_format="factor", case_number=10, complexity=5):
    # Ensure output_format is not "code" if FACTOR_CODES is intended to be used
    if output_format == "code":
        print("Warning: output_format is 'code' but FACTOR_CODES has been removed. Proceeding with factor names.")
        # output_format = "factor" # Optionally, force to "factor" or handle error

    all_cases = []
    # gen = ScenarioGenerator(mode1=mode, mode2=mode, output_format=output_format, complexity=complexity) # This line is moved inside the loop
    
    print(f"Generating {case_number} scenarios with mode={mode}, format={output_format}, complexity={complexity}")
    
    # Do mode mapping directly
    if mode in ScenarioGenerator.MODE_MAPPING:
        internal_mode = ScenarioGenerator.MODE_MAPPING[mode]
    elif mode in ScenarioGenerator.MODE_MAPPING.values():
        internal_mode = mode
    else:
        internal_mode = "unarguable"  # Default to unarguable if invalid
        
    print(f"Internal mode: {internal_mode}")
    
    # Generate the specified number of sets
    # scenario_sets = [] # Renamed to all_cases and initialized above
    for i in range(case_number):
        print(f"Generating scenario {i+1}/{case_number}")
        gen = ScenarioGenerator(mode1=internal_mode, mode2=internal_mode, output_format=output_format, complexity=complexity)
        
        # Verify that in unarguable mode there are no overlapping factors
        if internal_mode == "unarguable":
            max_attempts = 5
            attempt = 0
            while attempt < max_attempts:
                common_tsc1 = [f for f in gen.input_factors if f in gen.tsc1]
                common_tsc2 = [f for f in gen.input_factors if f in gen.tsc2]
                
                if common_tsc1 or common_tsc2:
                    print(f"Found overlaps after generation. TSC1: {len(common_tsc1)}, TSC2: {len(common_tsc2)}. Restarting ({attempt+1}/{max_attempts}).")
                    gen.restart()
                    attempt += 1
                else:
                    print("No overlaps found. Scenario is valid.")
                    break
                    
            if attempt >= max_attempts:
                print(f"Warning: Could not eliminate all overlaps after {max_attempts} attempts.")
        
        prompt = gen.generate_initial_prompt()
        print(f"Generated prompt for scenario {i+1}")
        all_cases.append(prompt)

    # Save to CSV
    filename_prefix = f"{internal_mode}_{output_format}_{case_number}_complexity{complexity}"
    print(f"Saving to {filename_prefix}.csv")
    
    try:
        with open(f"{filename_prefix}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario'])
            for scenario in all_cases:
                writer.writerow([scenario])
        print(f"Successfully saved to {filename_prefix}.csv")
    except IOError as e:
        print(f"Error saving dataset: {e}")

    # Save factor mapping to a separate JSON file
    # This part is removed as FACTOR_CODES is removed
    # try:
    #     with open(f"{filename_prefix}_factor_mapping.json", 'w') as f_map:
    #         json.dump(ScenarioGenerator.get_factor_mapping(), f_map, indent=4)
    #     print(f"Successfully saved factor mapping to {filename_prefix}_factor_mapping.json")
    # except IOError as e:
    #     print(f"Error saving factor mapping: {e}")
            
    return all_cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate legal case scenarios based on modes and complexity.")
    parser.add_argument("--mode", type=str, default="non-arguable", 
                        choices=list(ScenarioGenerator.MODE_MAPPING.keys()) + list(ScenarioGenerator.MODE_MAPPING.values()), 
                        help="Mode for scenario generation (non-arguable, mismatched, arguable).")
    parser.add_argument("--output_format", type=str, default="factor", choices=["factor", "code"], 
                        help="Output format for factors (factor or code).")
    parser.add_argument("--complexity", type=int, default=5, 
                        help="Complexity of the scenario (number of factors).")
    parser.add_argument("--num_cases", type=int, default=10, 
                        help="Number of cases to generate.")
    # parser.add_argument("--output_file", type=str, default="", 
    #                     help="Optional: Specify output CSV file name. If not provided, a default name will be used.")
    args = parser.parse_args()

    print(f"Starting scenario generator. Python version: {sys.version}")
    print(f"Arguments: mode={args.mode}, output_format={args.output_format}, num_cases={args.num_cases}, complexity={args.complexity}")

    if args.output_format == "code":
        # This check is now more of a notice since FACTOR_CODES is removed.
        # The actual handling of 'code' format (using factor names) is done in ScenarioGenerator and generate_datasets.
        print("Info: 'code' output format selected. Factor names will be used as FACTOR_CODES is removed.")
        print("No specific factor code mapping to display.")
        print("\n")

    # Generate datasets
    print(f"Generating {args.num_cases} case(s) with mode '{args.mode}', output format '{args.output_format}', and complexity {args.complexity}\n")
    
    try:
        datasets = generate_datasets(
            mode=args.mode,
            output_format=args.output_format,
            case_number=args.num_cases,
            complexity=args.complexity
        )
        
        # Save to CSV
        # Construct a default filename if output_file is not specified
        csv_file_path = f"{args.mode}_complexity{args.complexity}_{args.num_cases}cases_{args.output_format}.csv"
        # if args.output_file:
        #     csv_file_path = args.output_file
            
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Case #", "Input Scenario", "TSC1", "TSC2"]) # Header
            for i, data in enumerate(datasets):
                # Data is expected to be a dictionary from generate_initial_prompt
                # Ensure keys match what generate_initial_prompt returns after changes
                input_scenario_str = ";".join(data.get("input_scenario", []))
                tsc1_str = ";".join(data.get("tsc1", []))
                tsc2_str = ";".join(data.get("tsc2", []))
                writer.writerow([i + 1, input_scenario_str, tsc1_str, tsc2_str])
        
        print(f"Dataset saved to {csv_file_path}")
        print(f"Generated {len(datasets)} scenarios in '{args.mode}' mode with complexity {args.complexity}.")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()