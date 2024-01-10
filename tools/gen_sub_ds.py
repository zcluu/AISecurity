import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--origin_dir', type=str, required=True, help='Original dataset directory.(ImageNet or others)')
parser.add_argument('--target_dir', type=str, required=True, help='Target dataset directory.(Where to save)')
args = parser.parse_args()
os.makedirs(args.target_dir, exist_ok=True)

# Target Category, ImageNet or others
# Birds
dirs = [
    'n01530575',
    'n01531178',
    'n01532829',
    'n01534433',
    'n01537544',
    'n01558993',
    'n01560419',
    'n01580077',
    'n01582220',
    'n01592084'
]
for it in dirs:
    os.system(
        f'ln -s {os.path.join(args.origin_dir, it)} {os.path.join(args.target_dir, it)}"'
    )
