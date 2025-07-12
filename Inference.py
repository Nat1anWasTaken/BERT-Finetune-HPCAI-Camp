import torch
import time
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

def main(args):
    # load dataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_dataset = load_dataset("mteb/tweet_sentiment_extraction", split="test")

    print(f"length of dataset is {len(test_dataset)}")

    # inference
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, trust_remote_code=True,
                                                                low_cpu_mem_usage=True, device_map=device
                                                            )

    correct_count = 0
    start_time = time.time()

    with torch.no_grad():
        pbar = tqdm(range(len(test_dataset)))
        for i in pbar:
            # print(test_dataset[i]["text"])
            input_data = tokenizer(test_dataset[i]["text"], return_tensors="pt").to(device)

            for k, v in input_data.items():
                v = v.to(device)

            logits = model(**input_data).logits
            predicted_class_id = logits.argmax().item()

            if predicted_class_id == test_dataset[i]["label"]:
                correct_count += 1

    end_time = time.time()

    # print result
    print(f"The generation accuracy is {correct_count / len(test_dataset) * 100} %.")
    print(f"Total inference time is {end_time - start_time} sec.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased", help="The model name or the path to model's directory.")

    args = parser.parse_args()

    main(args)