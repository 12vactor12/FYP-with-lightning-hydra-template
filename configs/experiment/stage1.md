# phaseA,B
python src/train.py --multirun experiment=phaseAB data.train_subset_ratio=0.1,0.2,0.5 1 model.net.model_name=resnet50,vit_base_16,vit_base_16_dino seed=42,4189,92731


# phaseC_supcon
python src/train.py --multirun experiment=phaseC_supcon data.train_subset_ratio=0.1,0.2,0.5 1 model.net.model_name=vit_base_16_dino seed=42,4189,92731

# phaseC_probe(记得改ckpt)
python src/train.py --multirun experiment=phaseC_probe data.train_subset_ratio=0.1,0.2,0.5 1 model.net.model_name=vit_base_16_dino seed=42,4189,92731

# 恢复训练（对应的dir和version）
  save_dir: "${paths.output_dir}/tensorboard/"        
  version: 0     