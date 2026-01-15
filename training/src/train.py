import hydra
from omegaconf import DictConfig, OmegaConf
from Trainer import Model
import os
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.output_dir = cfg.output_dir + "_models"
    os.system("mkdir " + cfg.output_dir)
    cfg.train.output_dir = cfg.output_dir + "/"+ cfg.model_id.split("/")[1]
    print(OmegaConf.to_yaml(cfg))

    if "real" in cfg.output_dir:
        trainer = Model(cfg, real_eval=True)
    else:
        trainer = Model(cfg, real_eval=False)
    trainer.SFT()
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        print("Error occurred:", e)
        traceback.print_exc()
        sys.exit(1)