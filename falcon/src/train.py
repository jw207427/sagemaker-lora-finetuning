import os
# import nvidia

# cuda_install_dir = '/'.join(nvidia.__file__.split('/')[:-1]) + '/cuda_runtime/lib/'
# os.environ['LD_LIBRARY_PATH'] =  cuda_install_dir

import argparse
from pathlib import Path

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,     
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from random import randint 
from datasets import load_from_disk
import torch

from transformers import Trainer, TrainingArguments
from peft import PeftConfig, PeftModel
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-xl",
        # required=True,
        help="Model id to use for training.",
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
    
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/opt/ml/model", help="Model dir."
    )
    
#     parser.add_argument(
#         "--tensorboard_dir",
#         type=str,
#         default="/opt/ml/output/tensorboard",
#         help="Tensorboard dir.",
#     )
    
#     parser.add_argument("--log_steps", type=int, default=10, help="Log interval steps.")

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
    
def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    
    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
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


    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model

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
    # set seed
    set_seed(args.seed)
    
    # Get the parameters from the script arguments
    train_ds = load_from_disk(args.train_dataset_path)
    
    print(f"Total number of train samples: {len(train_ds)}")    
    
    test_ds = load_from_disk(args.test_dataset_path)
    
    print(f"Total number of test samples: {len(test_ds)}")    
    
    print(train_ds[randint(0, len(train_ds))])
    
    #load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                 trust_remote_code=True,
                                                 quantization_config=bnb_config,
                                                 device_map="auto")
    
    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    tokenizer.save_pretrained("/opt/ml/model/")
    
    # Set the Falcon tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # create peft config
    model = create_peft_config(model)
    
    output_dir = "/tmp"
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size, #8
        per_device_eval_batch_size=8,
        logging_dir=f"{output_dir}/logs",
        logging_steps=2,
        num_train_epochs=args.epochs, #1
        learning_rate=args.lr,        #2e-4
        bf16=args.bf16,  # Use BF16 if available True
        # logging strategies
        gradient_accumulation_steps=2,
        logging_strategy="steps",
        save_strategy="no",
        # optim="adafactor",
        output_dir=output_dir,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # Start training
    trainer.train()
    
    # merge adapter weights with base model and save
    # save int 8 model
    trainer.model.save_pretrained(output_dir)
    # clear memory
    del model 
    del trainer
    
    # load PEFT model in fp16
    peft_config = PeftConfig.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        trust_remote_code=True,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    model = PeftModel.from_pretrained(model, output_dir)
    model.eval()
    
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("/opt/ml/model/")
    
    # copy inference script
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "/opt/ml/model/code/requirements.txt",
    )
        
if __name__ == "__main__":
    args = parse_args()
    main(args)