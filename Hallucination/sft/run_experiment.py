"""
Unified experiment runner for monofact-targeted vs random upweighting.
Drop this file into the sft/ directory.

Usage:
    python run_experiment.py --condition baseline --alpha 1 --data_path ../data/biography_data.csv
    python run_experiment.py --condition random_upweight --alpha 1 --data_path ../data/biography_data.csv
    python run_experiment.py --condition monofact_upweight --alpha 1 --data_path ../data/biography_data.csv

Smoke test (quick 3-epoch run):
    python run_experiment.py --condition baseline --alpha 1 --data_path ../data/biography_data.csv --num_epochs 3
"""
import argparse
import math, copy, torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer
)
from datasets import Dataset, concatenate_datasets

from utils import (
    miscalibration_analysis, create_powerlaw_p,
    tokenize_function, custom_data_collator, sample,
    hallucination_analysis, inaccuracy_analysis
)
from utils_callback import CallBackTrainer
from select_subset import select_monofact_subset, select_random_subset


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()
    print(f"Device: {device} | fp16: {use_fp16}")

    # --- Reproducibility ---
    np.random.seed(1217)
    torch.manual_seed(1217)

    # --- Load data ---
    data = pd.read_csv(args.data_path)
    length_data = 10000
    dataset = data[0:length_data]

    train_datasets = {}
    p_datasets = {}
    for alpha in [1, 1.5, 2]:
        powerlaw_p = create_powerlaw_p(dataset, alpha)
        training_data = sample(powerlaw_p, length_data)
        train_datasets[alpha] = training_data
        p_datasets[alpha] = powerlaw_p

    # --- Load model ---
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- Select alpha ---
    alpha = args.alpha
    train_texts = train_datasets[alpha]
    print(f"\nCondition: {args.condition} | Alpha: {alpha} | Model: {args.model_name}")
    print(f"Monofact rate in training data: {len([1 for c in pd.Series(train_texts['y']).value_counts().values if c == 1]) / len(train_texts):.3f}")

    # --- Build HF Dataset ---
    train_dataset = Dataset.from_dict({
        "x": train_texts["x"].tolist(),
        "y": train_texts["y"].tolist(),
        "names": train_texts["names"].tolist(),
        "gold": train_texts["gold"].tolist()
    })
    tokenized_train = train_dataset.map(
        lambda ex: tokenize_function(tokenizer, ex), batched=False
    )
    split_dataset = tokenized_train.train_test_split(test_size=0.1, seed=1217)
    train_dataset_final = split_dataset["train"]
    eval_dataset_final = split_dataset["test"]

    # --- Output paths ---
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{args.condition}_alpha{alpha}_{args.model_name.replace('/', '_')}_metrics.csv"

    callback = CallBackTrainer(
        train_dataset_texts=train_dataset_final,
        train_datasets=train_datasets,
        p_datasets=p_datasets,
        tokenizer=tokenizer,
        alpha=alpha,
        device=device,
        output_csv_path=csv_path,
        epsilon=0.1,
        batch_size=48
    )

    collator = lambda features: custom_data_collator(features, tokenizer)

    # ========== CONDITION: BASELINE (no upweighting) ==========
    if args.condition == "baseline":
        baseline_epochs = args.num_epochs if args.num_epochs else 95
        training_args = TrainingArguments(
            output_dir=f"cache/{args.condition}_alpha{alpha}",
            num_train_epochs=baseline_epochs,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=32,
            learning_rate=5e-4,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="epoch",
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset_final,
            eval_dataset=eval_dataset_final,
            data_collator=collator,
            args=training_args,
            callbacks=[callback]
        )
        trainer.train()

    # ========== CONDITION: RANDOM or MONOFACT UPWEIGHT ==========
    elif args.condition in ("random_upweight", "monofact_upweight"):
        phase_epochs = args.num_epochs if args.num_epochs else 64

        # Phase 1: normal training
        phase1_args = TrainingArguments(
            output_dir=f"cache/{args.condition}_alpha{alpha}_phase1",
            num_train_epochs=phase_epochs,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=32,
            learning_rate=5e-4,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="epoch",
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset_final,
            eval_dataset=eval_dataset_final,
            data_collator=collator,
            args=phase1_args,
            callbacks=[callback]
        )
        trainer.train()

        # Phase 2: upweight selected subset
        if args.condition == "random_upweight":
            indices = select_random_subset(
                train_dataset_final,
                subset_fraction=args.subset_fraction,
                seed=1217
            )
        else:  # monofact_upweight
            indices = select_monofact_subset(
                train_dataset_final,
                subset_fraction=args.subset_fraction,
                seed=1217
            )

        train_subset = train_dataset_final.select(indices)
        train_subset_duplicated = concatenate_datasets(
            [train_subset] * args.duplications
        )
        print(f"\nPhase 2: upweighting {len(train_subset)} examples × {args.duplications} = {len(train_subset_duplicated)} rows")

        phase2_args = TrainingArguments(
            output_dir=f"cache/{args.condition}_alpha{alpha}_phase2",
            num_train_epochs=phase_epochs,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=32,
            learning_rate=5e-4,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="epoch",
            report_to="none"
        )
        trainer_dup = Trainer(
            model=model,
            train_dataset=train_subset_duplicated,
            eval_dataset=eval_dataset_final,
            data_collator=collator,
            args=phase2_args,
            callbacks=[callback]
        )
        trainer_dup.train()

    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    # ========== FINAL EVALUATION ==========
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        tvd, regret, miscal_table = miscalibration_analysis(
            p_datasets, train_datasets, model, tokenizer, alpha, 0.1, device
        )
        hall_rate = hallucination_analysis(
            model, train_dataset_final, tokenizer, 48, alpha, device
        )
        _, inacc_rate = inaccuracy_analysis(
            model, train_dataset_final, tokenizer, 48, alpha, device
        )

    print(f"\n{'=' * 50}")
    print(f"FINAL RESULTS — {args.condition} (alpha={alpha})")
    print(f"{'=' * 50}")
    print(f"  Hallucination rate:  {hall_rate:.4f}")
    print(f"  Miscalibration (TV): {tvd:.4f}")
    print(f"  KL divergence:       {regret:.4f}")
    print(f"  Inaccuracy:          {inacc_rate:.4f}")

    # Save final results
    final_results = pd.DataFrame([{
        "condition": args.condition,
        "alpha": alpha,
        "model": args.model_name,
        "hallucination_rate": hall_rate,
        "miscalibration_tv": tvd,
        "kl_divergence": regret,
        "inaccuracy": inacc_rate,
        "subset_fraction": args.subset_fraction,
        "duplications": args.duplications
    }])
    final_path = f"results/{args.condition}_alpha{alpha}_{args.model_name.replace('/', '_')}_final.csv"
    final_results.to_csv(final_path, index=False)
    print(f"Final results saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True,
                        choices=["baseline", "random_upweight", "monofact_upweight"])
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Pareto alpha for frequency skew (1, 1.5, or 2)")
    parser.add_argument("--data_path", type=str, default="../data/biography_data.csv")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--subset_fraction", type=float, default=0.05,
                        help="Fraction of training data to upweight")
    parser.add_argument("--duplications", type=int, default=10,
                        help="How many times to duplicate the upweighted subset")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override epoch count (for smoke testing). "
                             "Default: 95 for baseline, 64 per phase for upweight.")
    args = parser.parse_args()
    run_experiment(args)
