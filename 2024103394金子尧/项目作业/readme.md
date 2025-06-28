本项目代码使用扩散模型对抗体的6个CDR环进行有模板的设计。
Property-guided design of antibodies
 配置文件位于 configs/test 文件夹中。要分别设计 6 个 CDR，请在脚本 design_pdb.py上指定 config 为codesign_single 模型。
  
  Property-unconditioned design
  python design_pdb.py ./data/examples/7DK2_AB_C.pdb \
    --config ./configs/test/codesign_single.yml
