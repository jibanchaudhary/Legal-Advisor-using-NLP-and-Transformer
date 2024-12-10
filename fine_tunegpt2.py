from transformers import GPT2LMHeadModel,GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

def load_gpt2_model():
    model_name = "gpt2"
    print("Loading GPT-2 model ")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset(file_path, tokenizer, block_size = 128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer= tokenizer,
        mlm = False,
    )

def fine_tune_gpt2(dataset_path, output_dir, num_epochs):
    model, tokenizer = load_gpt2_model()
    dataset = load_dataset(dataset_path, tokenizer)
    data_collator = create_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        save_steps= 500,
        save_total_limit = 2,
        logging_dir=f"{output_dir}/logs",
        logging_steps = 100,
    )
    trainer = Trainer(
        model = model,
        args= training_args,
        train_dataset = dataset,
        data_collator= data_collator,
    )
    print("Starting fine_tuning")
    trainer.train()
    print("Saving model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved")

if __name__=="__main__":
    dataset_path = "dialogs.txt"
    output_dir = "./fine_tuned_gpt"
    num_epochs = 20
    fine_tune_gpt2(dataset_path, output_dir, num_epochs)