import os

import pandas as pd

IMG_DIR = './data/images'

def main():

    assocs = []

    tag_dirs = os.listdir(IMG_DIR)
    for tag_dir in tag_dirs:
        tag_dir_path = '/'.join([IMG_DIR, tag_dir])
        if os.path.isdir(tag_dir_path):
            images = os.listdir(tag_dir_path)
            for image_name in images:
                assocs.append((tag_dir, image_name))

    assocs_df = (
        pd.DataFrame(assocs, columns = ['tag', 'image'])
    )

    assocs_df.to_csv('./data/assocs.csv', sep = ',',
                     mode = 'w')

    assocs_pivot = (
        assocs_df
        .pivot_table(index=['image'], columns=['tag'],
                     aggfunc=[len], fill_value = 0)

    )

    assocs_pivot.to_csv('./data/assocs_pivot.csv', sep = ',',
                        mode = 'w')


main()
