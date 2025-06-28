基于可开发性属性指导的抗体序列与结构协同设计扩散模型

本项目代码使用扩散模型对抗体的6个CDR环进行有模板的设计。

Property-guided design of antibodies
 配置文件位于 configs/test 文件夹中。要分别设计 6 个 CDR，请在脚本 design_pdb.py上指定 config 为codesign_single 模型。

Property-unconditioned design
无条件属性设计
  python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
    --config ./configs/test/codesign_single.yml

Guiding design: Property-aware prior
对于 hydropathy-aware prior with b ，将选项 --prior_b 指定为0.8：
 python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
    --config ./configs/test/codesign_single.yml --prior_b 0.8

Guiding design: Sampling by property
指导性设计：按属性抽样
要按属性（ddG、水疗法或两者）进行采样，请使用以下配置文件：
一个 CDR 的序列结构， 按 ddG 采样 ：codesign_single_ddg.yml
一个 CDR 的序列结构， 按水疗法采样: codesign_single_hydro.yml
一个 CDR 的序列结构， 通过 ddG 和水疗法采样: codesign_single_ddg_and_hydro.yml
此处的额外选项为：--sample_step_mode （“min”、“max” 或 “softmax”）、--sample_step_num （int） 和 --sample_step_period （int）。例如，要按 ddG 进行采样，请使用：
  python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
    --config ./configs/test/codesign_single_ddg.yml \
    --sample_step_mode min --sample_step_num 20 --sample_step_period 1

Evaluation  评估
要计算评估指标：AAR、RMSD、Hydropathy Score 和 Predicted ddG（选项 --no_energy 阻止计算 Rosetta ddG）对于所有样本，请使用： 
python eval.py --no_energy --root results/codesign_single
