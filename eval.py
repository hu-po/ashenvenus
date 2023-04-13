from src import eval_from_episode_dir

eval_from_episode_dir(
    eval_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split\\valid",
    episode_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output\\34fdec50",
    output_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output\\pred_34fdec50",
    weights_filename = "model.pth",
    device='gpu',
    eval_on = '1',
    threshold = 0.3,
    max_num_samples_eval = 80000,
    max_time_hours = 0.5,
    batch_size = 6,
    log_images = False,
    save_pred_img = True,
    save_submit_csv = False,
    save_histograms = False,
)