from trainer import GedTrainer

def main():
    trainer = GedTrainer(dataset_path="data/AIDS700", input_dim=10)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()