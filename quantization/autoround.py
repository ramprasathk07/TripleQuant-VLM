from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
# from llmcompressor.modifiers.autoround import AutoRoundModifier
from .base_quantizer import BaseQuantizer

class AutoRoundQuantizer(BaseQuantizer):
	def quantize(self, dataset):
		recipe = [
			QuantizationModifier(
				ignore=self.config['autoround'].get('ignore', []),
				config_groups={
					"group_0": {
						"targets": self.config['autoround'].get('targets', ["Linear"]),
						"weights": {
							"num_bits": self.config['autoround']['bits'],
							"type": "int",
							"group_size": self.config['autoround']['group_size'],
							"strategy": "group",
						},
					}
				},
			)
		]

		proc_or_tok = self.processor if self.processor else self.tokenizer

		oneshot(
			model=self.model,
			dataset=dataset,
			processor=self.processor if self.processor else self.tokenizer,
			recipe=recipe,
			max_seq_length=self.config['calibration']['max_seq_length'],
			num_calibration_samples=self.config['calibration']['num_samples'],
		)
