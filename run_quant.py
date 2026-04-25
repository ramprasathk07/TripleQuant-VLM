import argparse
import os
from utils.config_parser import load_config
from quantization import AWQQuantizer, GPTQQuantizer, AutoRoundQuantizer

from datasets import load_dataset

def get_quantizer(config):
    method = config.get('method', 'awq').lower()
    if method == 'awq':
        return AWQQuantizer(config)
    elif method == 'gptq':
        return GPTQQuantizer(config)
    elif method == 'smooth_quant':
        return GPTQQuantizer(config)
    elif method == 'autoround':
        return AutoRoundQuantizer(config)
    # Add other methods here
    raise ValueError(f"Unsupported quantization method: {method}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs='+', required=True, help="Path(s) to config files (e.g. model, method)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. Initialize Quantizer
    quantizer = get_quantizer(config)
    
    # 2. Load Model
    quantizer.load_model()
    
    # 3. Prepare Dataset (VLM-aware)
    print(f"Loading dataset: {config['calibration']['dataset_id']}")
    from data.processors import get_vlm_dataset
    dataset = get_vlm_dataset(
        config['calibration']['dataset_id'],
        quantizer.processor,
        split=config['calibration']['dataset_split'],
        num_samples=config['calibration']['num_samples']
    )
    
    # 4. Run Quantization & Save
    save_name = config['model']['model_id'].split('/')[-1] + config['output']['save_dir_suffix']
    output_dir = os.path.join(config['output']['base_dir'], save_name)
    quantizer.quantize(dataset, output_dir=output_dir)
    
    # 5. Save processor/tokenizer separately if oneshot didn't
    quantizer.save(output_dir, weights=False)

if __name__ == "__main__":
    main()
