from src import eval_from_episode_dir

eval_from_episode_dir(
    episode_dir='/home/tren/dev/ashenvenus/output/6b230419',
    output_dir='/home/tren/dev/ashenvenus/preds',
    quantize=False,
    device='gpu',
    resize_ratio=1.0,
    batch_size=64,
)