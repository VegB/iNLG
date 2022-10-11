"""
Download instructions & image features for Diagnose-VLN.
"""
import os
import shutil
import sys
import urllib.request
import zipfile
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--override', action='store_true')
parser.add_argument('--download_data', type=int, default=1)
parser.add_argument('--download_checkpoint', type=int, default=1)
args = parser.parse_args()


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f'Create directory: {dir}')
    else:
        print(f'Exists: {dir}')


def _progress_hook(count, block_size, total_size):
    percent = float(count * block_size) / float(total_size) * 100.
    sys.stdout.write(f'\r>> Downloading ... {percent:.1f}%')
    sys.stdout.flush()


def download(content, item_list, dst_dir):
    print(f'Will download {content} for:\t{item_list}')
    check_dir(dst_dir)

    for item in item_list:
        print(f'Start processing {item}')

        cur_dst_dir = os.path.join(dst_dir, item)
        if os.path.isdir(cur_dst_dir):
            print(f'{cur_dst_dir} already exists.')
            if not args.override:
                print('Will skip to avoid override for now. If you want to override current content, please set `--override`.')
                continue
            else:
                print(f'Will override current content in {cur_dst_dir}')
                shutil.rmtree(cur_dst_dir)

        zip_filename = os.path.join(dst_dir, f'{item}.zip')
        url = f'https://inlg.s3.us-west-1.amazonaws.com/{content}/{item}.zip'
        urllib.request.urlretrieve(url, zip_filename, _progress_hook)
        print('\nExtracting ...')
        with zipfile.ZipFile(zip_filename) as zfile:
            zfile.extractall(dst_dir)
        os.remove(zip_filename)
        print(f'Successfully downloaded {content} for {item}. Extracted to {dst_dir}')



if __name__ == '__main__':
    # download data
    if args.download_data:
        download(
            content='data',
            item_list=['concept2text', 'sentence_completion', 'story_generation'],
            dst_dir='./data/'
        )

    # download checkpoint
    if args.download_checkpoint:
        download(
            content='checkpoint',
            item_list=['mapper', 'projection'],
            dst_dir='./checkpoint/'
        )
