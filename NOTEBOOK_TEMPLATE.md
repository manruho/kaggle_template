# Kaggle Notebook Template

```python
!git clone https://github.com/<you>/kaggle-template.git /kaggle/working/kaggle-template

import sys
sys.path.append("/kaggle/working/kaggle-template")

from src.config_io import load_config
from src.experiment import run

cfg = load_config("/kaggle/working/kaggle-template/configs/comp_example/train.json")
res = run(cfg)

print(res.scores)
```
