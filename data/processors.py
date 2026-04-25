from datasets import load_dataset

def get_vlm_dataset(dataset_id, processor, split="train", num_samples=64):
    """
    Loads and formats a dataset for VLM quantization.
    Ensures the output has 'text' and 'image' columns that llmcompressor can process.
    """
    ds = load_dataset(dataset_id, split=split)
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))
    
    def transform_fn(example):
        # Create the standard message format for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": "Transcribe the text in this image."},
                ],
            }
        ]
        # Apply chat template to get the text prompt
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # We can also pre-process images here to save memory if needed,
        # but processor.apply_chat_template handles metadata.
        # To strictly limit VRAM, we tell the processor to use fewer pixels
        return {
            "text": prompt,
            "image": example["image"]
        }

    # Map the dataset and remove original columns to avoid conflicts
    return ds.map(transform_fn, remove_columns=ds.column_names)
