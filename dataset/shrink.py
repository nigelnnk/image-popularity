import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_PATH = 'data/reddit_data.csv'
MAX_RESOLUTION = 1024


def main(data_path, max_resolution):
    data = pd.read_csv(data_path)
    num_copied = 0
    num_resized = 0
    for path in tqdm(data['PATH']):
        path = Path(path)
        parts = list(path.parts)
        parts[1] = f'{parts[1]}_{max_resolution}'
        shrink_path = Path(*parts)
        shrink_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(path) as im:
            if im.width <= max_resolution and im.height <= max_resolution:
                shutil.copy(path, shrink_path)
                num_copied += 1
            else:
                im.thumbnail((max_resolution, max_resolution))
                im.save(shrink_path, 'JPEG', quality=50, optimize=True)
                num_resized += 1
    print(f'Copied: {num_copied}, Resized: {num_resized}')


if __name__ == '__main__':
    main(DATA_PATH, MAX_RESOLUTION)
