from src import eval_from_episode_dir

eval_from_episode_dir(
    eval_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split\\valid",
    episode_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output\\083f0bda",
    output_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output\\pred_083f0bda",
    weights_filename = "model_083f0bda_best_1.pth",
    device='gpu',
    eval_on = '1',
    threshold = 0.3,
    max_num_samples_eval = 1000,
    max_time_hours = 0.008,
    batch_size = 2,
    log_images = True,
    save_pred_img = True,
    save_submit_csv = True,
    save_histograms = True,
)