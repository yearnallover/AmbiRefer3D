import hydra
import lightning.pytorch as pl

from m3drefclip.data.data_module import DataModule


@hydra.main(version_base=None, config_path="config", config_name="global_config_FRONT")
def main(cfg):
    # 固定随机种子（可选）
    pl.seed_everything(cfg.test_seed, workers=True)

    # datamodule（会按 cfg.data 读 val split）
    dm = DataModule(cfg.data)

    # 构建模型（Lightning 会在 validate 时用 ckpt 覆盖权重）
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)
    dm.setup(stage="fit")
    # 关 logger，避免 wandb / tensorboard 干扰；只看终端输出
    trainer = pl.Trainer(logger=False, **cfg.trainer)

    # 跑验证并打印指标
    results = trainer.validate(model=model, datamodule=dm, ckpt_path=cfg.ckpt_path)
    print("\n[VALIDATE RESULTS]")
    for r in results:
        for k, v in r.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
