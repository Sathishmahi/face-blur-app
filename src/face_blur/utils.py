# Import necessary libraries
import os
import yaml  # This line imports the PyYAML library for working with YAML files

# Define a function 'read_config' for reading configuration settings from a YAML file
def read_config(config_file: str = os.path.join("config", "config.yaml")) -> dict:
    # Check if the configuration file exists
    if not os.path.exists(config_file):
        # Raise a FileNotFoundError if the file doesn't exist
        raise FileNotFoundError(f"config file not found {config_file}")
    
    try:
        # Open the configuration file for reading ('r' mode)
        with open(config_file, 'r') as f:
            # Load the contents of the YAML file into a Python dictionary
            config = yaml.safe_load(f)
        # Return the loaded configuration as a dictionary
        return config
    except yaml.YAMLError as exc:
        # If there is an error in parsing the YAML file, raise the exception
        raise (exc)
