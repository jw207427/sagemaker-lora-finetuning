import argparse
import json
import datasets
from pathlib import Path
import os
import nvidia

import torch
from accelerate import Accelerator
from accelerate.utils import LoggerType
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import transformers

cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        # required=True,
        help="Path to pretrained sent-transformer model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/opt/ml/input/data/train",
        # required=True,
        help="Path to the train dataset.",
    )

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="/opt/ml/input/data/test",
        # required=True,
        help="Path to the test dataset.",
    )

    parser.add_argument(
        "--cat_encoders_path",
        type=str,
        default="/opt/ml/input/data/encoders",
        # required=True,
        help="Path to the category encoders.",
    )
    
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    
    # LoRA config
    parser.add_argument("--lora_alpha", type=int, default=16, help="Gain factor")
    parser.add_argument("--lora_dropout", type=int, default=0.1, help="LoRA Dropout")
    parser.add_argument("--lora_r", type=int, default=64, help="rank of LoRA parameters")
    
    # trainer config
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per instance")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate steps over multiple batch for training stability")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer")
    parser.add_argument("--save_steps", type=int, default=10, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--learning_rate", type=int, default=2e-4, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=int, default=0.3, help="Max gradiant normal")
    parser.add_argument("--max_steps", type=int, default=500, help="Max iteration")
    parser.add_argument("--warmup_ratio", type=int, default=0.03, help="Warm ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate schedule")

    parser.add_argument(
        "--model_dir", type=str, default="/opt/ml/model", help="Model dir."
    )
    
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="/opt/ml/output/tensorboard",
        help="Tensorboard dir.",
    )
    
    parser.add_argument("--log_steps", type=int, default=10, help="Log interval steps.")

    args = parser.parse_args()

    return args

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main(args):
    
    # Get the parameters from the script arguments
    train_path = Path(args.train_dataset_path)
    test_path = Path(args.test_dataset_path)
    
    tb_log_dir = args.tensorboard_dir
    tb_log_interval = args.log_steps

    model_id = args.pretrained_model_name_or_path
    
    accelerator = Accelerator(log_with=LoggerType.TENSORBOARD, project_dir=tb_log_dir)
    
    accelerator.init_trackers(".", init_kwargs={"tensorboard": {"flush_secs": 30}})

    # load the datasets 
    train_ds = datasets.Dataset.from_json((train_path / "train.json").as_posix())
    
    test_ds = datasets.Dataset.from_json((test_path / "test.json").as_posix())
    

    #load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 trust_remote_code=True, 
                                                 quantization_config=bnb_config,
                                                 device_map="auto")
    model.config.use_cache = False

    # load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # tokenize and chunk dataset
    lm_train_dataset = train_ds.map(
        lambda sample: tokenizer(sample["text"]), batched=True, batch_size=24, remove_columns=list(train_ds.features)
    )


    lm_test_dataset = test_ds.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(test_ds.features)
    )

    # Print total number of samples
    print(f"Total number of train samples: {len(lm_train_dataset)}")    
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Below we will load the configuration file in order to create the LoRA model. According to QLoRA paper, it is important     # to consider all linear layers in the transformer block for maximum performance. Therefore we will add `dense`, 
    # `dense_h_to_4_h` and `dense_4h_to_h` layers in the target modules in addition to the mixed query key value layer.
    
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
            ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    # Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a         
    # wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters.   
    # Let's first load the training arguments below.
    # bucket = "sagemaker-us-west-2-376678947624"
    # log_bucket = f"s3://{bucket}/falcon-7b-qlora-finetune"

    trainer = transformers.Trainer(
        model=model,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # logging_dir=log_bucket,
            # logging_steps=2,
            num_train_epochs=1,
            learning_rate=2e-4,
            gradient_accumulation_steps=4,
            bf16=True,
            save_strategy = "no",
            # output_dir="outputs",
            # report_to="tensorboard",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                            mlm=False),
    )
    
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False  
    # Start training
    trainer.train()
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Evalute metrics
        metrics = trainer.evaluate
        accelerator.print(metrics)
        accelerator.log(metrics, step=total_steps)
        

        # Save the model to the output folder
        trainer.save_model(args.model_dir)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()    
    print("Training complete......")

        
if __name__ == "__main__":
    args = parse_args()
    main(args)