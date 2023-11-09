from typing import List, Dict
import json
import numpy as np
import math

from metrics import word_f1, bleu

from agents.user_config import UserAgent


def load_json_data(file_path: str, keys: List[str]) -> List[Dict]:
    with open(file_path, "r") as fp:
        data = json.load(fp)

    result = []
    for dialogue in data:
        updated_dialogue = {}
        for turn_id, sample in dialogue.items():
            if not isinstance(sample, dict):
                continue            
            sample_data = {key: sample[key] for key in keys}
            updated_dialogue[turn_id] = sample_data
        result.append(updated_dialogue)
    return result


def load_data(file_path: str) -> List[Dict]:
    # NOTE to participants: Gold reference will not available during actual evaluations
    keys = ["persona A", "persona B", "dialogue", "gold_reference"] 
    return load_json_data(file_path, keys)


def get_responses(agent, test_data, BATCH_SIZE):
    all_responses = [{} for _ in range(len(test_data))]
    split_size = math.ceil(len(test_data) / BATCH_SIZE)
    for batch_idx in np.array_split(range(len(test_data)), split_size):
        for turn_id in range(7):
            batch_inputs = [test_data[i][f"turn_{turn_id}"] for i in batch_idx]
            responses = agent.generate_responses(batch_inputs)
            for resp in responses:
                for bi in batch_idx:
                    all_responses[bi][f"turn_{turn_id}"] = resp
    return all_responses

def evaluate(responses, test_data):
    f1_scores = []
    bleu_scores = []
    for response, test_data_single in zip(responses, test_data):
        for turn_id in range(7):
            f1 = word_f1(response[f"turn_{turn_id}"],
                         [test_data_single[f"turn_{turn_id}"]['gold_reference']])
            bleu_score = bleu(response[f"turn_{turn_id}"],
                         [test_data_single[f"turn_{turn_id}"]['gold_reference']])
            f1_scores.append(f1)
            bleu_scores.append(bleu_score)
    return np.mean(f1_scores), np.mean(bleu_scores)

if __name__ == "__main__":
    BATCH_SIZE = 2
    data_path = 'dummy_data_task1.json'
    test_data = load_data(data_path)
    agent = UserAgent()
    responses = get_responses(agent, test_data, BATCH_SIZE)
    f1_score, bleu_score = evaluate(responses, test_data)

    print("Word F1 Score:", f1_score)
    print("Word Bleu Score:", bleu_score)

