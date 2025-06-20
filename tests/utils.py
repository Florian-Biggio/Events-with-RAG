import json

def load_expected_metrics(metrics_file: str):
    """
    Load the expected migration metrics from a JSON file.

    Args:
        metrics_file (str): Path to the JSON file containing expected metrics.
    
    Returns:
        dict: A dictionary containing the metrics.
    """
    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics
