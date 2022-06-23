

from pathlib import Path
import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if not iskaggle:
    import zipfile,kaggle
    path = Path('feedback-prize-effectiveness')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

from fastai.imports import *

"""# Examining the data

Let's see what is available for us to work with for this competition:
"""

if iskaggle: path = Path('../input/feedback-prize-effectiveness')
path.ls()

