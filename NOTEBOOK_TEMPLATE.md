# Kaggle Notebook Template

```python
# clone
!rm -rf /kaggle/working/kaggle-template
!git clone https://github.com/manruho/kaggle_template /kaggle/working/kaggle-template

# import
import sys
sys.path.insert(0, "/kaggle/working/kaggle-template")

from src.config_io import load_config
from src.experiment import run

cfg = load_config("/kaggle/working/kaggle-template/configs/default.json")
result = run(cfg)
result
```
