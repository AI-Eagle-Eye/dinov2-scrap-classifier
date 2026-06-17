import subprocess

experiments = [
    ("exp_ab_mean_pad", "configs/exp_ab_mean_pad.yaml"),
    ("exp_ah_orig",     "configs/exp_ah_orig.yaml"),
    ("exp_ah_regularize", "configs/exp_ah_regularize.yaml"),
    ("exp_aj_linear",   "configs/exp_aj_linear.yaml"),
    ("exp_ak_regularize2", "configs/exp_ak_regularize2.yaml"),
    ("exp_al_ema",      "configs/exp_al_ema.yaml"),
    ("exp_am_cls_att",  "configs/exp_am_cls_att.yaml"),
    ("exp_an_tuned",    "configs/exp_an_tuned.yaml"),
    ("exp_ap_f1best",   "configs/exp_ap_f1best.yaml"),
    ("exp_aq_multipad", "configs/exp_aq_multipad.yaml"),
]

for exp, config in experiments:
    ckpt = f"experiments/{exp}/checkpoints/best_model.pth"
    print(f"\n{'='*50}\n{exp} 평가 중...\n{'='*50}")
    subprocess.run([
        "python", "scripts/evaluate_test.py",
        "--config", config,
        "--checkpoint", ckpt,
        "--testset_root", "dataset/testset",
    ])