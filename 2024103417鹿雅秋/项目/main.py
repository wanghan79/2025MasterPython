import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model.cot import CoTTrajectoryPredictor
from utils.data_processor import HighDDataProcessor
from tqdm import tqdm
import json
from datetime import datetime

def print_section(title, content="", char="=", length=80):
    print(f"\n{char * length}")
    print(f"【{title}】")
    if content:
        print(f"{content}")
    print(f"{char * length}\n")

def print_step(step_name, content="", char="-", length=50):
    print(f"\n{char * length}")
    print(f">> {step_name}")
    if content:
        print(f"{content}")
    print(f"{char * length}\n")
class TrajectoryPredictionTrainer:
    def __init__(self, data_path, processed_data_path=None, batch_size=32, 
                 learning_rate=0.001, num_epochs=50, device=None):
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize data processor
        self.data_processor = HighDDataProcessor(
            data_path=data_path,
            processed_data_path=processed_data_path
        )
        
        # Initialize model
        self.model = CoTTrajectoryPredictor(device=self.device)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"Initialized trainer with device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def prepare_data(self, recording_ids):
        """Prepare data for training"""
        all_train_data = []
        all_val_data = []
        
        for recording_id in recording_ids:
            try:
                # Load processed data
                data = self.data_processor.load_processed_data(recording_id)
                
                # Split into train and validation
                split_idx = int(0.8 * len(data['sequence_X']))
                
                train_data = {
                    'X': data['sequence_X'][:split_idx],
                    'y': data['sequence_y'][:split_idx]
                }
                val_data = {
                    'X': data['sequence_X'][split_idx:],
                    'y': data['sequence_y'][split_idx:]
                }
                
                all_train_data.append(train_data)
                all_val_data.append(val_data)
                
                print(f"Loaded recording {recording_id}")
                print(f"Train samples: {len(train_data['X'])}")
                print(f"Validation samples: {len(val_data['X'])}")
                
            except Exception as e:
                print(f"Error loading recording {recording_id}: {str(e)}")
                continue
        
        # Combine all data
        train_X = np.concatenate([d['X'] for d in all_train_data], axis=0)
        train_y = np.concatenate([d['y'] for d in all_train_data], axis=0)
        val_X = np.concatenate([d['X'] for d in all_val_data], axis=0)
        val_y = np.concatenate([d['y'] for d in all_val_data], axis=0)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_X),
            torch.FloatTensor(train_y)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_X),
            torch.FloatTensor(val_y)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training", leave=True, ncols=100) as pbar:
            for batch_idx, (X, y) in enumerate(pbar):
                # Move data to device
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions, explanation = self.model(X)
                
                # Compute loss
                loss, loss_components = self.model.compute_loss(predictions, y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update total loss
                total_loss += loss.item()
                current_avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_description(f"Training [{batch_idx}/{num_batches}]")
                pbar.set_postfix_str(
                    f"loss: {loss.item():.4f}, avg_loss: {current_avg_loss:.4f}"
                )
                
                # Log batch metrics
                if batch_idx % 100 == 0:
                    print(f"\nBatch {batch_idx}/{num_batches}")
                    print(f"Current loss: {loss.item():.4f}")
                    print(f"Average loss: {current_avg_loss:.4f}")
                    print("Loss components:")
                    for k, v in loss_components.items():
                        print(f"  {k}: {v:.4f}")
                    if explanation:
                        print(f"Sample explanation: {explanation}")
        
        epoch_avg_loss = total_loss / num_batches
        print(f"\nEpoch completed. Average loss: {epoch_avg_loss:.4f}")
        
        return epoch_avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc="Validating", leave=False):
                # Move data to device
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                predictions, _ = self.model(X)
                
                # Compute loss
                loss, _ = self.model.compute_loss(predictions, y)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_predictions.append({k: v.cpu().numpy() for k, v in predictions.items()})
                all_targets.append(y.cpu().numpy())
        
        # Compute average loss
        avg_loss = total_loss / len(val_loader)
        
        # Compute additional metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def compute_metrics(self, predictions, targets):
        """Compute evaluation metrics"""
        metrics = {}
        
        # Combine all predictions and targets
        combined_predictions = {
            k: np.concatenate([p[k] for p in predictions], axis=0)
            for k in predictions[0].keys()
        }
        combined_targets = np.concatenate(targets, axis=0)
        
        # Compute metrics for each prediction horizon
        for term in ['short_term', 'mid_term', 'long_term']:
            pred = combined_predictions[term]
            if term == 'short_term':
                target = combined_targets[:, :4]
            elif term == 'mid_term':
                target = combined_targets[:, :6]
            else:
                target = combined_targets
            
            # Position error
            pos_error = np.mean(np.sqrt(
                (pred[:, :, 0] - target[:, :, 0])**2 +
                (pred[:, :, 1] - target[:, :, 1])**2
            ))
            
            # Velocity error
            vel_error = np.mean(np.sqrt(
                (pred[:, :, 2] - target[:, :, 2])**2 +
                (pred[:, :, 3] - target[:, :, 3])**2
            ))
            
            metrics[f'{term}_position_error'] = float(pos_error)
            metrics[f'{term}_velocity_error'] = float(vel_error)
        
        return metrics
    
    def train(self, recording_ids):
        """Train the model"""
        train_loader, val_loader = self.prepare_data(recording_ids)
        best_model_path = 'best_model.pth'
        start_time = datetime.now()
        
        print_section("Training Configuration", 
                     f"Total epochs: {self.num_epochs}\n"
                     f"Batch size: {self.batch_size}\n"
                     f"Learning rate: {self.learning_rate}")
        
        for epoch in range(self.num_epochs):
            print_section(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print_step("Training Results", f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            print_step("Validation Results", 
                      f"Validation loss: {val_loss:.4f}\n"
                      f"Metrics:\n{json.dumps(val_metrics, indent=2)}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, best_model_path)
            
            # Early stopping check
            if epoch > 10 and val_loss > min(self.val_losses[:-10]):
                print("Early stopping triggered")
                break
        
        # Training summary
        duration = datetime.now() - start_time
        print("\nTraining completed!")
        print(f"Total training time: {duration}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'training_time': str(duration)
        }

def main():
    # 配置参数
    config = {
        'data_path': 'HighD/data',
        'processed_data_path': 'HighD/processed_data',
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'recording_ids': list(range(1, 23))  # 修改为1-22
    }

    # 初始化训练器
    trainer = TrajectoryPredictionTrainer(
        data_path=config['data_path'],
        processed_data_path=config['processed_data_path'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs']
    )

    # 开始训练
    print("Starting training...")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    results = trainer.train(config['recording_ids'])
    
    # 保存结果
    save_dir = os.path.join('results', datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {save_dir}")

    # 2. 准备LLM训练数据
    llm_training_data = []
    for recording_id in config['recording_ids']:
        data = trainer.data_processor.load_processed_data(recording_id)
        # 生成训练数据对
        for trajectory in data['sequence_X']:
            prompt = trainer.model._generate_prompt_from_trajectory(trajectory)
            target = trainer.model._generate_target_from_trajectory(trajectory)
            llm_training_data.append((prompt, target))
    
    # 3. 训练LLM
    trainer.model.train_llm(
        train_data=llm_training_data,
        num_epochs=3,
        save_dir="llm_checkpoints"
    )

if __name__ == "__main__":
    main() 