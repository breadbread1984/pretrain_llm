#!/usr/bin/python3

from absl import flags, app
import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('pretrained_ckpt', default = None, help = 'path to output checkpoint of pretraining')
  flags.DEFINE_string('sft_ckpt', default = 'sft', help = 'path to output checkpoint of sft')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_float('lr', default = '2e-4', help = 'learning rate')

def main(unused_argv):
  dataset = load_dataset('json', data_files = FLAGS.dataset, split = "train")
  model = AutoModelForCausalLM.from_pretrained(FLAGS.pretrained_ckpt, attn_implementation = "flash_attention_2", trust_remote_code = True)
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_ckpt, trust_remote_code = True)
  model_training_args = TrainingArguments(
    output_dir = FLAGS.sft_ckpt,
    per_device_train_batch_size = 4,
    optim = "adamw_torch",
    logging_steps = 80,
    learning_rate = FLAGS.lr,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    num_train_epochs = 1,
    save_strategy = "epoch")
  
