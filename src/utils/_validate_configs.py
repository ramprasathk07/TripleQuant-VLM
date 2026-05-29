import glob, yaml, os
from src.config.schemas import QuantizeConfig
from src.quantization.llm_compressor import LLMCompressorQuantizer

ok = True
for path in sorted(glob.glob('config/quantize/**/*.yaml', recursive=True)):
    rel = os.path.relpath(path, 'config/quantize').replace(os.sep, '/')
    try:
        cfg = QuantizeConfig.model_validate(yaml.safe_load(open(path, encoding='utf-8')))
        extra = ''
        if cfg.backend == 'llm_compressor':
            rec = LLMCompressorQuantizer(cfg)._build_recipe()
            extra = ' ' + str([type(m).__name__ for m in rec])
        print(f"OK   {rel:46s} {cfg.backend:14s} {cfg.method:5s} {cfg.scheme.scheme:12s}{extra}")
    except Exception as e:
        ok = False
        print(f"FAIL {rel}: {type(e).__name__}: {str(e)[:90]}")
print('RESULT', 'ALL PASS' if ok else 'HAS FAILURES')
