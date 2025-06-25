import os
import argparse
import torch
from data import ProteinDataset, ToGraphTransform, split_dataset
from model import VAE
from train import Trainer
from generate_eval import Generator, Evaluator
from utils import load_pdb
from config import set_seed, DEVICE, DATA_DIR, MODELS_DIR, PDB_OUT_DIR, PLOT_DIR, NPY_OUT_DIR, logger

def parse_args():
    parser = argparse.ArgumentParser(description='Protein Conformation Generation')
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train / generate / eval / analyze', required=True)

    # 训练模式
    train_parser = subparsers.add_parser('train', help='Train VAE model')
    train_parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='数据目录')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    train_parser.add_argument('--epochs', type=int, default=EPOCHS)
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    train_parser.add_argument('--output_dir', type=str, default=MODELS_DIR)

    # 生成模式
    gen_parser = subparsers.add_parser('generate', help='Generate conformations')
    gen_parser.add_argument('--model_path', type=str, required=True, help='已训练模型路径')
    gen_parser.add_argument('--seq_file', type=str, required=True, help='用于生成的序列FASTA文件')
    gen_parser.add_argument('--num', type=int, default=10, help='生成样本数量')

    # 评估模式
    eval_parser = subparsers.add_parser('eval', help='Evaluate generated conformations')
    eval_parser.add_argument('--true_coords_dir', type=str, required=True, help='真实坐标目录(.npy)')
    eval_parser.add_argument('--gen_coords_dir', type=str, required=True, help='生成坐标目录(.npy)')
    eval_parser.add_argument('--no_align', action='store_true', help='不进行 Kabsch 对齐')

    # 分析模式
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results: plot saved graphs')
    analyze_parser.add_argument('--plot_type', type=str, choices=['loss', 'rmsd'], required=True, help='选择要查看的图: loss 或 rmsd')

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed()

    if args.mode == 'train':
        full_dataset = ProteinDataset(args.data_dir, transform=ToGraphTransform())
        train_dataset, val_dataset = split_dataset(full_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        seq_len_example = train_dataset[0]['seq_feat'].size(0)
        model = VAE(seq_length=seq_len_example)
        trainer = Trainer(model, train_loader, val_loader, lr=args.lr, epochs=args.epochs, output_dir=args.output_dir)
        trainer.train()

    elif args.mode == 'generate':
        tmp_seq_dataset = ProteinDataset(os.path.dirname(args.seq_file))
        seq_feat = tmp_seq_dataset._load_sequence(args.seq_file)
        model = VAE(seq_length=seq_feat.size(0))
        generator = Generator(model, args.model_path, seq_feat)
        coords_list = generator.generate(args.num)
        print(f"Generated {len(coords_list)} samples and saved to {PDB_OUT_DIR} and {NPY_OUT_DIR}")

    elif args.mode == 'eval':
        true_coords_list = []
        gen_coords_list = []
        for tf in sorted(os.listdir(args.true_coords_dir)):
            true_path = os.path.join(args.true_coords_dir, tf)
            true_coords = torch.tensor(np.load(true_path), dtype=torch.float32)
            true_coords_list.append(true_coords)
        for gf in sorted(os.listdir(args.gen_coords_dir)):
            gen_path = os.path.join(args.gen_coords_dir, gf)
            gen_coords = torch.tensor(np.load(gen_path), dtype=torch.float32)
            gen_coords_list.append(gen_coords)
            
        evaluator = Evaluator(true_coords_list)
        stats = evaluator.evaluate(gen_coords_list, align=(not args.no_align))
        print("Evaluation Results:")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")

    elif args.mode == 'analyze':
        if args.plot_type == 'loss':
            loss_curve_path = os.path.join(PLOT_DIR, 'loss_curve.png')
            if os.path.exists(loss_curve_path):
                from PIL import Image
                img = Image.open(loss_curve_path)
                img.show()
            else:
                print(f"No loss curve found at {loss_curve_path}")
        elif args.plot_type == 'rmsd':
            rmsd_hist_path = os.path.join(PLOT_DIR, 'rmsd_histogram.png')
            if os.path.exists(rmsd_hist_path):
                from PIL import Image
                img = Image.open(rmsd_hist_path)
                img.show()
            else:
                print(f"No RMSD histogram found at {rmsd_hist_path}")

    else:
        print("Unsupported mode. Use 'train', 'generate', 'eval', or 'analyze'.")

if __name__ == '__main__':
    main()
