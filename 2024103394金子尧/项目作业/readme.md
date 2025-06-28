# Antibody Co-Design with Property-Guided Diffusion Models

本项目提供了一种基于**可开发性属性指导的扩散模型**，用于**抗体序列与结构的协同设计**。支持对抗体的 6 个 CDR 区域进行有模板或属性引导的设计。


## 项目结构

* `design_pdb.py：主设计脚本
* `eval.py：评估设计结果
* `configs/test/：不同设计策略对应的配置文件
* `data/examples/：输入结构示例（如 `7DK2_AB_C.pdb`）

## 使用方法

### 1. Property-Guided Design（属性引导的设计）

使用模型设计一个或多个 CDR 区域，具备对特定可开发性属性的控制能力。

python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
  --config ./configs/test/codesign_single.yml


### 2. Property-Unconditioned Design（无属性引导的设计）

直接生成设计，不指定任何属性约束。

python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
  --config ./configs/test/codesign_single.yml

### 3. Guiding Design: Property-aware Prior（属性感知先验引导）

以 Hydropathy-aware prior 为例，指定 `--prior_b` 参数：

python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
  --config ./configs/test/codesign_single.yml \
  --prior_b 0.8


### 4. Guiding Design: Sampling by Property（按属性采样）

根据 ddG、水疗性（Hydropathy），或两者进行引导采样：

| 采样属性             | 配置文件                                |
| ---------------- | ----------------------------------- |
| ddG              | `codesign_single_ddg.yml`           |
| Hydropathy       | `codesign_single_hydro.yml`         |
| ddG + Hydropathy | `codesign_single_ddg_and_hydro.yml` |

使用示例（以 ddG 为例）：

python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
  --config ./configs/test/codesign_single_ddg.yml \
  --sample_step_mode min \
  --sample_step_num 20 \
  --sample_step_period 1

参数说明：
--sample_step_mode`: `min`, `max`, `softmax`
--sample_step_num`: 抽样次数
--sample_step_period`: 采样周期

## 评估指标

评估设计样本的结构质量与功能相关性，包括：

AAR (Antigen-Antibody RMSD)
RMSD
Hydropathy Score
Predicted ddG

默认会计算 Rosetta ddG。若不需要，可加上 `--no_energy`：

python eval.py --no_energy --root results/codesign_single


## 示例数据

项目已提供 PDB 格式的输入结构示例：

data/examples/7DK2_AB_C.pdb


## 备注

* 仅支持**一个 CDR 区域的设计**（如 Loop H3），多个 CDR 需分别指定。
* 推荐使用 GPU 加速以提升生成效率。
* 输出将保存在 `results/ 目录中，按配置名称分类。


