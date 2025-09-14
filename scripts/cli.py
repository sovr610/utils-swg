#!/usr/bin/env python3
"""
Liquid-Spiking Neural Network CLI Application

This CLI tool provides a complete interface for training, saving, loading, 
and running the hybrid liquid-spiking neural networks defined in main.py.

Usage:
    python cli.py train --task vision --epochs 20
    python cli.py load --model-path vision_model.pt --input-file test_image.npy
    python cli.py benchmark --model-path vision_model.pt
    python cli.py export --model-path vision_model.pt --format onnx
"""

import argparse
import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import platform
import psutil
import subprocess

# Rich imports for beautiful CLI
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
)
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.tree import Tree
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.markdown import Markdown
from rich import print as rprint
from rich.status import Status

# Import from main.py
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import (
    TaskType, ModelConfig, LiquidSpikingNetwork, LiquidSpikingTrainer,
    DatasetFactory, create_llm_config, create_vision_config, create_robotics_config,
    create_custom_config, save_config, load_config, print_config_summary,
    get_model_parameter_count, load_model, benchmark_model, export_onnx, 
    generate_text, evaluate_perplexity, train_llm_model, train_vision_model, 
    train_robotics_model
)

# Initialize Rich console
console = Console()

class RichLogger:
    """Enhanced logger using Rich for beautiful output."""
    
    def __init__(self):
        self.console = console
    
    def info(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [green]ℹ[/green] {message}")
    
    def warning(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [yellow]⚠[/yellow] {message}")
    
    def error(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [red]✗[/red] {message}")
    
    def success(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [green]✓[/green] {message}")
    
    def header(self, title: str, subtitle: str = ""):
        """Create a beautiful header panel."""
        if subtitle:
            content = f"[bold blue]{title}[/bold blue]\n[dim]{subtitle}[/dim]"
        else:
            content = f"[bold blue]{title}[/bold blue]"
        
        panel = Panel(
            content,
            border_style="blue",
            padding=(1, 2),
            title="🧠 Liquid-Spiking Neural Network",
            title_align="left"
        )
        self.console.print(panel)

class SystemInfo:
    """Gather and display system information."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Collect comprehensive system information."""
        info = {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor() or "Unknown",
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').total,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else 0,
            })
        
        # Check for specific dependencies
        try:
            import snntorch
            info['snntorch_version'] = snntorch.__version__
        except ImportError:
            info['snntorch_version'] = "Not installed"
        
        try:
            import ncps
            info['ncps_version'] = "Available"
        except ImportError:
            info['ncps_version'] = "Not installed"
        
        return info

class LiquidSpikingCLI:
    """Enhanced CLI application with Rich graphics."""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.logger = RichLogger()
        self.console = console
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            description="🧠 Liquid-Spiking Neural Network CLI Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
[bold blue]Examples:[/bold blue]
  [green]# Train a vision model for 20 epochs[/green]
  python cli.py train --task vision --epochs 20 --batch-size 64
  
  [green]# Train with multi-GPU acceleration (auto-detect GPUs)[/green]
  python cli.py train --task llm --epochs 15 --multi-gpu
  
  [green]# Train with specific GPUs using DataParallel[/green]
  python cli.py train --task vision --gpu-strategy dp --gpu-ids "0,1,2"
  
  [green]# Train with DistributedDataParallel (recommended for 4+ GPUs)[/green]
  python cli.py train --task robotics --gpu-strategy ddp --multi-gpu
  
  [green]# Load and run inference on an image[/green]
  python cli.py inference --model-path vision_model.pt --input-file test.npy
  
  [green]# Benchmark a trained model[/green]
  python cli.py benchmark --model-path vision_model.pt --iterations 1000
  
  [green]# Export model to ONNX format[/green]
  python cli.py export --model-path vision_model.pt --output-path model.onnx
  
  [green]# Interactive configuration editor[/green]
  python cli.py config --task robotics --interactive
  
  [green]# Check system and GPU information[/green]
  python cli.py info --system --gpu
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        self._add_train_parser(subparsers)
        
        # Inference command
        self._add_inference_parser(subparsers)
        
        # Benchmark command
        self._add_benchmark_parser(subparsers)
        
        # Export command
        self._add_export_parser(subparsers)
        
        # Config command
        self._add_config_parser(subparsers)
        
        # Info command
        self._add_info_parser(subparsers)
        
        # Status command
        self._add_status_parser(subparsers)
        
        return parser
    
    def _add_train_parser(self, subparsers):
        """Add training subcommand parser."""
        train_parser = subparsers.add_parser('train', help='🚀 Train a neural network model')
        
        # Basic training parameters
        train_parser.add_argument('--task', choices=['llm', 'vision', 'robotics'], 
                                required=True, help='Type of task to train for')
        train_parser.add_argument('--epochs', type=int, default=10, 
                                help='Number of training epochs (default: 10)')
        train_parser.add_argument('--batch-size', type=int, 
                                help='Batch size for training (overrides config default)')
        train_parser.add_argument('--learning-rate', type=float,
                                help='Learning rate (overrides config default)')
        
        # Configuration file options
        train_parser.add_argument('--config-path', type=str,
                                help='Path to custom configuration JSON file')
        train_parser.add_argument('--save-config', type=str,
                                help='Save final configuration to JSON file')
        
        # Neural network architecture parameters
        arch_group = train_parser.add_argument_group('Neural Network Architecture')
        arch_group.add_argument('--liquid-units', type=int,
                               help='Number of liquid neural network units')
        arch_group.add_argument('--spiking-units', type=int,
                               help='Number of spiking neural network units')
        arch_group.add_argument('--num-layers', type=int,
                               help='Number of hybrid liquid-spiking layers')
        arch_group.add_argument('--hidden-dim', type=int,
                               help='Hidden dimension size')
        arch_group.add_argument('--num-attention-heads', type=int,
                               help='Number of attention heads')
        arch_group.add_argument('--liquid-backbone', choices=['cfc', 'ltc', 'ncp'],
                               help='Liquid neural network backbone type')
        
        # Spiking network parameters
        spike_group = train_parser.add_argument_group('Spiking Network Parameters')
        spike_group.add_argument('--spike-threshold', type=float,
                                help='Spike threshold for spiking neurons')
        spike_group.add_argument('--beta', type=float,
                                help='Membrane potential decay factor (0-1)')
        spike_group.add_argument('--num-spike-steps', type=int,
                                help='Number of time steps for spiking dynamics')
        
        # LLM-specific parameters
        llm_group = train_parser.add_argument_group('Language Model Parameters')
        llm_group.add_argument('--vocab-size', type=int,
                              help='Vocabulary size for LLM (default: 50257)')
        llm_group.add_argument('--embedding-dim', type=int,
                              help='Embedding dimension for tokens')
        llm_group.add_argument('--max-position-embeddings', type=int,
                              help='Maximum position embeddings')
        llm_group.add_argument('--sequence-length', type=int,
                              help='Maximum sequence length')
        
        # Vision-specific parameters
        vision_group = train_parser.add_argument_group('Vision Model Parameters')
        vision_group.add_argument('--conv-channels', type=str,
                                 help='Convolutional channels (comma-separated, e.g., "32,64,128")')
        vision_group.add_argument('--conv-kernel-sizes', type=str,
                                 help='Convolutional kernel sizes (comma-separated, e.g., "3,3,3")')
        
        # Regularization parameters
        reg_group = train_parser.add_argument_group('Regularization')
        reg_group.add_argument('--dropout', type=float,
                              help='Dropout rate')
        reg_group.add_argument('--attention-dropout', type=float,
                              help='Attention dropout rate')
        reg_group.add_argument('--embedding-dropout', type=float,
                              help='Embedding dropout rate')
        reg_group.add_argument('--weight-decay', type=float,
                              help='Weight decay (L2 regularization)')
        reg_group.add_argument('--gradient-clip', type=float,
                              help='Gradient clipping value')
        
        # Training options
        train_parser.add_argument('--output-dir', type=str, default='./models',
                                help='Directory to save trained models (default: ./models)')
        train_parser.add_argument('--resume', type=str,
                                help='Path to checkpoint to resume training from')
        train_parser.add_argument('--save-interval', type=int, default=5,
                                help='Save checkpoint every N epochs (default: 5)')
        train_parser.add_argument('--no-validation', action='store_true',
                                help='Skip validation during training')
        train_parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                                help='Device to use for training (default: auto)')
        train_parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', default=True,
                                help='Enable mixed precision training (default: True)')
        train_parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false',
                                help='Disable mixed precision training')
        train_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducibility')
        
        # Multi-GPU training options
        gpu_group = train_parser.add_argument_group('Multi-GPU Training')
        gpu_group.add_argument('--multi-gpu', action='store_true',
                              help='Enable multi-GPU training (auto-detect available GPUs)')
        gpu_group.add_argument('--gpu-strategy', choices=['auto', 'dp', 'ddp', 'none'], 
                              default='auto',
                              help='Multi-GPU strategy: auto (automatic), dp (DataParallel), ddp (DistributedDataParallel), none (single GPU/CPU)')
        gpu_group.add_argument('--gpu-ids', type=str,
                              help='Specific GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
        gpu_group.add_argument('--distributed-backend', choices=['nccl', 'gloo'], 
                              default='nccl',
                              help='Distributed training backend (nccl for GPU, gloo for CPU)')
        gpu_group.add_argument('--master-port', type=str, default='12355',
                              help='Port for distributed training communication (default: 12355)')
        gpu_group.add_argument('--sync-batchnorm', action='store_true', default=True,
                              help='Use synchronized batch normalization for distributed training')
        gpu_group.add_argument('--no-sync-batchnorm', action='store_false', dest='sync_batchnorm',
                              help='Disable synchronized batch normalization')
    
    def _add_inference_parser(self, subparsers):
        """Add inference subcommand parser."""
        inference_parser = subparsers.add_parser('inference', help='🔮 Run inference on trained model')
        inference_parser.add_argument('--model-path', type=str, required=True,
                                    help='Path to trained model checkpoint')
        inference_parser.add_argument('--input-file', type=str,
                                    help='Path to input data file (.npy, .pt, or .json)')
        inference_parser.add_argument('--input-shape', type=str,
                                    help='Shape of random input data (e.g., "3,32,32" for CIFAR-10)')
        inference_parser.add_argument('--batch-size', type=int, default=1,
                                    help='Batch size for inference (default: 1)')
        inference_parser.add_argument('--output-file', type=str,
                                    help='Path to save inference results')
        inference_parser.add_argument('--verbose', action='store_true',
                                    help='Print detailed inference results')
    
    def _add_benchmark_parser(self, subparsers):
        """Add benchmark subcommand parser."""
        benchmark_parser = subparsers.add_parser('benchmark', help='⚡ Benchmark model performance')
        benchmark_parser.add_argument('--model-path', type=str, required=True,
                                    help='Path to trained model checkpoint')
        benchmark_parser.add_argument('--iterations', type=int, default=100,
                                    help='Number of inference iterations (default: 100)')
        benchmark_parser.add_argument('--warmup', type=int, default=10,
                                    help='Number of warmup iterations (default: 10)')
        benchmark_parser.add_argument('--batch-sizes', type=str, default='1,8,16,32',
                                    help='Comma-separated batch sizes to test (default: 1,8,16,32)')
        benchmark_parser.add_argument('--output-file', type=str,
                                    help='Path to save benchmark results as JSON')
    
    def _add_export_parser(self, subparsers):
        """Add export subcommand parser."""
        export_parser = subparsers.add_parser('export', help='📦 Export trained model to different formats')
        export_parser.add_argument('--model-path', type=str, required=True,
                                 help='Path to trained model checkpoint')
        export_parser.add_argument('--output-path', type=str, required=True,
                                 help='Path for exported model')
        export_parser.add_argument('--format', choices=['onnx', 'torchscript'], default='onnx',
                                 help='Export format (default: onnx)')
        export_parser.add_argument('--opset-version', type=int, default=11,
                                 help='ONNX opset version (default: 11)')
    
    def _add_config_parser(self, subparsers):
        """Add config subcommand parser."""
        config_parser = subparsers.add_parser('config', help='⚙️ Create or modify model configurations')
        config_parser.add_argument('--task', choices=['llm', 'vision', 'robotics'], required=True,
                                 help='Type of task configuration')
        config_parser.add_argument('--save-path', type=str, required=True,
                                 help='Path to save configuration JSON file')
        config_parser.add_argument('--modify', type=str,
                                 help='JSON string with configuration modifications')
        config_parser.add_argument('--interactive', action='store_true',
                                 help='Interactive configuration editor')
    
    def _add_info_parser(self, subparsers):
        """Add info subcommand parser."""
        info_parser = subparsers.add_parser('info', help='ℹ️ Display model and system information')
        info_parser.add_argument('--model-path', type=str,
                               help='Path to model checkpoint to analyze')
        info_parser.add_argument('--system', action='store_true',
                               help='Display system information')
        info_parser.add_argument('--gpu', action='store_true',
                               help='Display detailed GPU information for multi-GPU training')
        info_parser.add_argument('--config-only', action='store_true',
                               help='Display only configuration information')
    
    def _add_status_parser(self, subparsers):
        """Add status subcommand parser."""
        status_parser = subparsers.add_parser('status', help='📊 Show training status and model overview')
        status_parser.add_argument('--models-dir', type=str, default='./models',
                                 help='Directory containing model checkpoints')
        status_parser.add_argument('--watch', action='store_true',
                                 help='Continuously monitor training status')
    
    def run(self):
        """Main entry point for the CLI application."""
        args = self.parser.parse_args()
        
        # Show welcome message
        self._show_welcome()
        
        if not args.command:
            self._show_help_menu()
            return
        
        try:
            if args.command == 'train':
                self._handle_train(args)
            elif args.command == 'inference':
                self._handle_inference(args)
            elif args.command == 'benchmark':
                self._handle_benchmark(args)
            elif args.command == 'export':
                self._handle_export(args)
            elif args.command == 'config':
                self._handle_config(args)
            elif args.command == 'info':
                self._handle_info(args)
            elif args.command == 'status':
                self._handle_status(args)
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠ Operation interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n[red]✗ An error occurred: {str(e)}[/red]")
            console.print_exception()
            sys.exit(1)
    
    def _show_welcome(self):
        """Display welcome message."""
        welcome_text = """
[bold blue]Liquid-Spiking Neural Network CLI[/bold blue]
[dim]Hybrid architecture combining liquid and spiking dynamics[/dim]

[green]Available Commands:[/green]
• [cyan]train[/cyan]     - Train neural network models
• [cyan]inference[/cyan] - Run model inference
• [cyan]benchmark[/cyan] - Performance benchmarking
• [cyan]export[/cyan]    - Export models to different formats
• [cyan]config[/cyan]    - Configuration management
• [cyan]info[/cyan]      - System and model information
• [cyan]status[/cyan]    - Training status monitoring
        """
        
        panel = Panel(
            welcome_text,
            border_style="blue",
            padding=(1, 2),
            title="🧠 Welcome",
            title_align="left"
        )
        console.print(panel)
    
    def _show_help_menu(self):
        """Show interactive help menu."""
        console.print("\n[yellow]💡 Use --help with any command for detailed options[/yellow]")
        console.print("[dim]Example: python cli.py train --help[/dim]\n")
    
    def _handle_train(self, args):
        """Handle the train command with rich progress display and multi-GPU support."""
        self.logger.header("Training Neural Network", f"Task: {args.task.upper()}")
        
        # Display GPU information if multi-GPU is enabled
        if hasattr(args, 'multi_gpu') and (args.multi_gpu or args.gpu_strategy != 'none'):
            self._display_gpu_info()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create configuration
        config = self._load_config(args)
        
        # Display configuration table
        self._display_config_table(config)
        
        # Create datasets with progress
        with Status("[bold green]Creating datasets...", spinner="dots"):
            train_dataset, val_dataset = self._create_datasets(config, args)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(config, args, train_dataset, val_dataset)
        
        # Initialize model and trainer
        with Status("[bold green]Initializing model and multi-GPU setup...", spinner="dots"):
            model = LiquidSpikingNetwork(config)
            trainer = LiquidSpikingTrainer(model, config)
        
        # Display model information
        self._display_model_info(model)
        
        # Display multi-GPU training info
        if hasattr(trainer, 'gpu_ids') and trainer.gpu_ids and len(trainer.gpu_ids) > 1:
            self._display_multi_gpu_info(trainer)
        
        # Training loop with rich progress bar
        self._train_with_progress(trainer, train_loader, val_loader, args, output_dir)
    
    def _train_with_progress(self, trainer, train_loader, val_loader, args, output_dir):
        """Train model with beautiful progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            
            # Main training progress
            train_task = progress.add_task(
                "[green]Training Progress", 
                total=args.epochs
            )
            
            # Epoch details
            epoch_task = progress.add_task(
                "[blue]Current Epoch", 
                total=len(train_loader)
            )
            
            start_time = time.time()
            
            for epoch in range(args.epochs):
                epoch_start = time.time()
                
                # Update epoch progress
                progress.update(
                    epoch_task, 
                    description=f"[blue]Epoch {epoch+1}/{args.epochs}",
                    completed=0
                )
                
                # Training epoch with batch progress
                train_loss = self._train_epoch_with_progress(
                    trainer, train_loader, progress, epoch_task
                )
                
                # Validation
                val_loss = None
                if val_loader is not None:
                    with Status("[yellow]Running validation...", spinner="dots"):
                        is_best = trainer.validate(val_loader)
                        val_loss = trainer.val_losses[-1]
                        
                        if is_best:
                            best_path = output_dir / f"{args.task}_best_model.pt"
                            trainer.save_checkpoint(str(best_path))
                            self.logger.success(f"New best model saved: {best_path}")
                
                epoch_time = time.time() - epoch_start
                
                # Update training progress
                progress.update(train_task, advance=1)
                
                # Log epoch results
                self._log_epoch_results(epoch, args.epochs, train_loss, val_loss, epoch_time)
                
                # Save periodic checkpoint
                if (epoch + 1) % args.save_interval == 0:
                    checkpoint_path = output_dir / f"{args.task}_epoch_{epoch+1}.pt"
                    trainer.save_checkpoint(str(checkpoint_path))
            
            # Save final model
            final_path = output_dir / f"{args.task}_final_model.pt"
            trainer.save_checkpoint(str(final_path))
            
            total_time = time.time() - start_time
            self.logger.success(f"Training completed in {total_time/3600:.2f} hours")
            self.logger.success(f"Final model saved: {final_path}")
    
    def _train_epoch_with_progress(self, trainer, train_loader, progress, epoch_task):
        """Train single epoch with progress updates."""
        trainer.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data and targets from batch dictionary
            if isinstance(batch, dict):
                data = batch["input_ids"]
                targets = batch["labels"]
            else:
                # Fallback for tuple format
                data, targets = batch
            
            # Train single batch using trainer's built-in logic
            batch_loss = self._train_single_batch(trainer, data, targets)
            total_loss += batch_loss
            num_batches += 1
            
            # Update batch progress
            progress.update(epoch_task, advance=1)
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_single_batch(self, trainer, data, targets):
        """Train a single batch and return loss."""
        data = data.to(trainer.device, non_blocking=True)
        targets = targets.to(trainer.device, non_blocking=True)
        
        trainer.optimizer.zero_grad()
        
        if trainer.config.mixed_precision and trainer.scaler:
            with autocast('cuda'):
                outputs = trainer.model(data)
                loss = trainer._compute_loss(outputs, targets)
            
            trainer.scaler.scale(loss).backward()
            
            if trainer.config.gradient_clip > 0:
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.gradient_clip)
            
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            outputs = trainer.model(data)
            loss = trainer._compute_loss(outputs, targets)
            loss.backward()
            
            if trainer.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.gradient_clip)
            
            trainer.optimizer.step()
        
        return loss.item()
    
    def _log_epoch_results(self, epoch, total_epochs, train_loss, val_loss, epoch_time):
        """Log epoch results in a formatted table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Epoch", f"{epoch+1}/{total_epochs}")
        table.add_row("Train Loss", f"{train_loss:.4f}")
        if val_loss is not None:
            table.add_row("Val Loss", f"{val_loss:.4f}")
        table.add_row("Time", f"{epoch_time:.2f}s")
        
        console.print(table)
    
    def _handle_benchmark(self, args):
        """Handle benchmark command with rich tables."""
        self.logger.header("Performance Benchmark", f"Model: {Path(args.model_path).name}")
        
        # Load model
        with Status("[bold green]Loading model...", spinner="dots"):
            model, config = load_model(args.model_path, None)
        
        batch_sizes = list(map(int, args.batch_sizes.split(',')))
        results = {}
        
        # Create benchmark table
        table = Table(title="Performance Benchmark Results")
        table.add_column("Batch Size", justify="center", style="cyan")
        table.add_column("Avg Time (ms)", justify="right", style="green")
        table.add_column("Throughput (samples/s)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="magenta")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            benchmark_task = progress.add_task(
                "[green]Benchmarking...", 
                total=len(batch_sizes)
            )
            
            for batch_size in batch_sizes:
                progress.update(
                    benchmark_task,
                    description=f"[green]Batch size: {batch_size}"
                )
                
                # Run benchmark for this batch size
                result = self._benchmark_batch_size(model, config, batch_size, args)
                results[batch_size] = result
                
                # Add row to table
                table.add_row(
                    str(batch_size),
                    f"{result['avg_time_ms']:.2f}",
                    f"{result['throughput_samples_per_sec']:.1f}",
                    f"{result['memory_allocated_mb']:.1f}"
                )
                
                progress.update(benchmark_task, advance=1)
        
        # Display results
        console.print(table)
        
        # Model statistics
        self._display_model_stats(model, config)
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.success(f"Benchmark results saved to {args.output_file}")
    
    def _handle_info(self, args):
        """Handle info command with rich display."""
        self.logger.header("System & Model Information")
        
        if args.system or not args.model_path:
            self._display_system_info()
        
        if args.gpu:
            self._display_gpu_info()
        
        if args.model_path:
            self._display_model_info_detailed(args.model_path, args.config_only)
    
    def _handle_status(self, args):
        """Handle status command with model overview."""
        self.logger.header("Training Status & Model Overview")
        
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            self.logger.warning(f"Models directory not found: {models_dir}")
            return
        
        # Find all model checkpoints
        model_files = list(models_dir.glob("*.pt"))
        
        if not model_files:
            self.logger.warning(f"No model checkpoints found in {models_dir}")
            return
        
        # Create status table
        table = Table(title=f"Model Status - {models_dir}")
        table.add_column("Model", style="cyan")
        table.add_column("Task", style="green")
        table.add_column("Epochs", justify="right", style="yellow")
        table.add_column("Best Val Loss", justify="right", style="magenta")
        table.add_column("Parameters", justify="right", style="blue")
        table.add_column("Size", justify="right", style="red")
        
        for model_path in sorted(model_files):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                config = ModelConfig.from_dict(checkpoint['config'])
                
                # Calculate model size
                size_mb = model_path.stat().st_size / (1024 * 1024)
                
                # Get training info
                epochs = len(checkpoint.get('train_losses', [0]))
                best_val = checkpoint.get('best_val_loss', 'N/A')
                
                # Count parameters
                model = LiquidSpikingNetwork(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                params = sum(p.numel() for p in model.parameters())
                
                table.add_row(
                    model_path.name,
                    config.task_type.value,
                    str(epochs),
                    f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val),
                    f"{params:,}",
                    f"{size_mb:.1f} MB"
                )
                
            except Exception as e:
                table.add_row(
                    model_path.name,
                    "[red]Error[/red]",
                    "-", "-", "-", "-"
                )
        
        console.print(table)
    
    def _display_config_table(self, config):
        """Display configuration in a beautiful table."""
        table = Table(title="Model Configuration", box=None)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            if key == 'task_type':
                value = value.value if hasattr(value, 'value') else value
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
    
    def _display_model_info(self, model):
        """Display model information in a table."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        table = Table(title="Model Architecture", box=None)
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="white")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Model Size (approx)", f"{total_params * 4 / (1024**2):.1f} MB")
        
        console.print(table)
    
    def _display_system_info(self):
        """Display comprehensive system information including GPU details."""
        system_info = SystemInfo.get_system_info()
        
        # System table
        sys_table = Table(title="🖥️ System Information", show_header=False)
        sys_table.add_column("Property", style="cyan")
        sys_table.add_column("Value", style="white")
        
        sys_table.add_row("Platform", system_info['platform'])
        sys_table.add_row("Python", system_info['python_version'])
        sys_table.add_row("PyTorch", system_info['pytorch_version'])
        sys_table.add_row("CPU Cores", str(system_info['cpu_count']))
        sys_table.add_row("RAM", f"{system_info['memory_gb']:.1f} GB")
        
        # Dependencies table
        deps_table = Table(title="📦 Key Dependencies", show_header=False)
        deps_table.add_column("Package", style="green")
        deps_table.add_column("Version", style="white")
        
        for pkg, version in system_info['dependencies'].items():
            deps_table.add_row(pkg, version)
        
        # CUDA/GPU table
        if system_info['cuda_available']:
            cuda_table = Table(title="🔥 CUDA/GPU Information", show_header=False)
            cuda_table.add_column("Property", style="yellow")
            cuda_table.add_column("Value", style="white")
            
            cuda_table.add_row("CUDA Available", "✅ Yes")
            cuda_table.add_row("CUDA Version", system_info['cuda_version'])
            cuda_table.add_row("GPU Count", str(system_info['gpu_count']))
            cuda_table.add_row("Primary GPU", system_info['gpu_name'])
            cuda_table.add_row("GPU Memory", f"{system_info['gpu_memory_gb']:.1f} GB")
            
            console.print(Columns([sys_table, deps_table, cuda_table]))
        else:
            console.print(Columns([sys_table, deps_table]))
    
    def _display_gpu_info(self):
        """Display detailed GPU information for multi-GPU training."""
        # Import GPU utils
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils.gpu_utils import GPUDetector
        
        # Get GPU information
        gpus = GPUDetector.detect_gpus()
        
        if not gpus:
            console.print("🚫 [red]No GPUs detected or CUDA unavailable[/red]")
            return
        
        # Create GPU table
        gpu_table = Table(title="🔥 Available GPUs for Multi-GPU Training")
        gpu_table.add_column("ID", justify="center")
        gpu_table.add_column("Name", style="cyan")
        gpu_table.add_column("Memory", justify="right")
        gpu_table.add_column("Compute", justify="center")
        gpu_table.add_column("Status", justify="center")
        gpu_table.add_column("Temp", justify="right")
        gpu_table.add_column("Power", justify="right")
        
        for gpu in gpus:
            status = "✅" if gpu.is_available else "❌"
            memory = f"{gpu.memory_total / 1024:.1f} GB"
            compute = f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            temp = f"{gpu.temperature}°C" if gpu.temperature else "N/A"
            power = f"{gpu.power_usage}W" if gpu.power_usage else "N/A"
            
            gpu_table.add_row(
                str(gpu.device_id),
                gpu.name,
                memory,
                compute,
                status,
                temp,
                power
            )
        
        console.print(gpu_table)
        
        # Show compatible GPUs
        compatible_gpus = GPUDetector.filter_compatible_gpus(gpus)
        if compatible_gpus:
            console.print(f"\n✅ [green]{len(compatible_gpus)} GPU(s) available for training[/green]")
            gpu_ids = [str(gpu.device_id) for gpu in compatible_gpus]
            console.print(f"   GPU IDs: {', '.join(gpu_ids)}")
        else:
            console.print("\n⚠️ [yellow]No compatible GPUs found for training[/yellow]")
    
    def _display_multi_gpu_info(self, trainer):
        """Display multi-GPU training configuration."""
        if not hasattr(trainer, 'gpu_ids') or not trainer.gpu_ids:
            return
        
        # Multi-GPU info table
        gpu_info_table = Table(title="⚡ Multi-GPU Training Configuration")
        gpu_info_table.add_column("Setting", style="cyan")
        gpu_info_table.add_column("Value", style="white")
        
        strategy = getattr(trainer.config, 'multi_gpu_strategy', 'unknown')
        gpu_count = len(trainer.gpu_ids)
        gpu_ids_str = ', '.join(map(str, trainer.gpu_ids))
        world_size = getattr(trainer, 'world_size', gpu_count)
        
        gpu_info_table.add_row("Strategy", strategy.upper())
        gpu_info_table.add_row("GPU Count", str(gpu_count))
        gpu_info_table.add_row("GPU IDs", gpu_ids_str)
        gpu_info_table.add_row("World Size", str(world_size))
        gpu_info_table.add_row("Batch Size", str(trainer.config.batch_size))
        gpu_info_table.add_row("Distributed", "Yes" if trainer.multi_gpu_manager.is_distributed else "No")
        
        console.print(gpu_info_table)
        
        # Performance estimation
        base_batch_size = 32  # Assume base batch size
        speedup_estimate = min(gpu_count * 0.85, gpu_count)  # Account for overhead
        
        console.print(f"\n🚀 [green]Expected training speedup: ~{speedup_estimate:.1f}x[/green]")
        console.print(f"   📊 Effective batch size: {trainer.config.batch_size}")
        console.print(f"   🔥 Multi-GPU acceleration active!")
    
    def _display_system_info_old(self):
        """Display system information in a comprehensive table."""
        sys_info = SystemInfo.get_system_info()
        
        # System table
        sys_table = Table(title="System Information")
        sys_table.add_column("Component", style="cyan")
        sys_table.add_column("Details", style="white")
        
        sys_table.add_row("Platform", sys_info['platform'])
        sys_table.add_row("Architecture", sys_info['architecture'])
        sys_table.add_row("Processor", sys_info['processor'])
        sys_table.add_row("Python Version", sys_info['python_version'])
        sys_table.add_row("CPU Cores", f"{sys_info['cpu_count']} ({sys_info['cpu_count_physical']} physical)")
        sys_table.add_row("Memory", f"{sys_info['memory_total'] / (1024**3):.1f} GB total, {sys_info['memory_available'] / (1024**3):.1f} GB available")
        
        # Dependencies table
        deps_table = Table(title="Dependencies")
        deps_table.add_column("Package", style="cyan")
        deps_table.add_column("Version", style="white")
        
        deps_table.add_row("PyTorch", sys_info['pytorch_version'])
        deps_table.add_row("snnTorch", sys_info['snntorch_version'])
        deps_table.add_row("NCPS", sys_info['ncps_version'])
        
        # CUDA table if available
        if sys_info['cuda_available']:
            cuda_table = Table(title="CUDA Information")
            cuda_table.add_column("Component", style="cyan")
            cuda_table.add_column("Details", style="white")
            
            cuda_table.add_row("CUDA Version", sys_info.get('cuda_version', 'Unknown'))
            cuda_table.add_row("Device Count", str(sys_info['cuda_device_count']))
            cuda_table.add_row("Device Name", sys_info.get('cuda_device_name', 'Unknown'))
            cuda_table.add_row("Memory", f"{sys_info.get('cuda_memory_total', 0) / (1024**3):.1f} GB")
            
            # Print all tables
            console.print(Columns([sys_table, deps_table, cuda_table]))
        else:
            console.print(Columns([sys_table, deps_table]))
    
    def _load_config(self, args):
        """Load configuration with user feedback and command-line overrides."""
        # Load base configuration
        if args.config_path:
            self.logger.info(f"Loading configuration from {args.config_path}")
            config = load_config(args.config_path)
        else:
            if args.task == 'llm':
                config = create_llm_config()
            elif args.task == 'vision':
                config = create_vision_config()
            else:
                config = create_robotics_config()
        
        # Override configuration with command line arguments
        overrides = {}
        
        # Basic training parameters
        if args.batch_size:
            overrides['batch_size'] = args.batch_size
        if args.learning_rate:
            overrides['learning_rate'] = args.learning_rate
        if args.epochs:
            overrides['num_epochs'] = args.epochs
        if hasattr(args, 'seed') and args.seed:
            overrides['seed'] = args.seed
        
        # Neural network architecture parameters
        if hasattr(args, 'liquid_units') and args.liquid_units:
            overrides['liquid_units'] = args.liquid_units
        if hasattr(args, 'spiking_units') and args.spiking_units:
            overrides['spiking_units'] = args.spiking_units
        if hasattr(args, 'num_layers') and args.num_layers:
            overrides['num_layers'] = args.num_layers
        if hasattr(args, 'hidden_dim') and args.hidden_dim:
            overrides['hidden_dim'] = args.hidden_dim
        if hasattr(args, 'num_attention_heads') and args.num_attention_heads:
            overrides['num_attention_heads'] = args.num_attention_heads
        if hasattr(args, 'liquid_backbone') and args.liquid_backbone:
            overrides['liquid_backbone'] = args.liquid_backbone
        
        # Spiking network parameters
        if hasattr(args, 'spike_threshold') and args.spike_threshold:
            overrides['spike_threshold'] = args.spike_threshold
        if hasattr(args, 'beta') and args.beta:
            overrides['beta'] = args.beta
        if hasattr(args, 'num_spike_steps') and args.num_spike_steps:
            overrides['num_spike_steps'] = args.num_spike_steps
        
        # LLM-specific parameters
        if hasattr(args, 'vocab_size') and args.vocab_size:
            overrides['vocab_size'] = args.vocab_size
        if hasattr(args, 'embedding_dim') and args.embedding_dim:
            overrides['embedding_dim'] = args.embedding_dim
        if hasattr(args, 'max_position_embeddings') and args.max_position_embeddings:
            overrides['max_position_embeddings'] = args.max_position_embeddings
        if hasattr(args, 'sequence_length') and args.sequence_length:
            overrides['sequence_length'] = args.sequence_length
        
        # Vision-specific parameters
        if hasattr(args, 'conv_channels') and args.conv_channels:
            overrides['conv_channels'] = [int(x.strip()) for x in args.conv_channels.split(',')]
        if hasattr(args, 'conv_kernel_sizes') and args.conv_kernel_sizes:
            overrides['conv_kernel_sizes'] = [int(x.strip()) for x in args.conv_kernel_sizes.split(',')]
        
        # Regularization parameters
        if hasattr(args, 'dropout') and args.dropout is not None:
            overrides['dropout'] = args.dropout
        if hasattr(args, 'attention_dropout') and args.attention_dropout is not None:
            overrides['attention_dropout'] = args.attention_dropout
        if hasattr(args, 'embedding_dropout') and args.embedding_dropout is not None:
            overrides['embedding_dropout'] = args.embedding_dropout
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            overrides['weight_decay'] = args.weight_decay
        if hasattr(args, 'gradient_clip') and args.gradient_clip is not None:
            overrides['gradient_clip'] = args.gradient_clip
        
        # Training options
        if hasattr(args, 'mixed_precision') and args.mixed_precision:
            overrides['mixed_precision'] = True
            
        # Device handling
        if args.device != 'auto':
            overrides['device'] = args.device
        elif args.device == 'auto':
            overrides['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Multi-GPU parameters
        if hasattr(args, 'multi_gpu') and args.multi_gpu:
            overrides['multi_gpu_strategy'] = 'auto'
        if hasattr(args, 'gpu_strategy') and args.gpu_strategy:
            overrides['multi_gpu_strategy'] = args.gpu_strategy
        if hasattr(args, 'gpu_ids') and args.gpu_ids:
            # Parse comma-separated GPU IDs
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            overrides['gpu_ids'] = gpu_ids
        if hasattr(args, 'distributed_backend') and args.distributed_backend:
            overrides['distributed_backend'] = args.distributed_backend
        if hasattr(args, 'sync_batchnorm') and args.sync_batchnorm is not None:
            overrides['sync_batchnorm'] = args.sync_batchnorm
        
        # Apply overrides using create_custom_config
        if overrides:
            self.logger.info(f"Applying {len(overrides)} configuration overrides")
            # Remove task_type from base config to avoid conflict
            base_config = config.to_dict()
            base_config.pop('task_type', None)
            config = create_custom_config(args.task, **{**base_config, **overrides})
        
        # Display configuration summary
        print_config_summary(config)
        
        # Show parameter count
        params = get_model_parameter_count(config)
        self.logger.info(f"Estimated model parameters: {params['total']:,}")
        
        # Save configuration if requested
        if hasattr(args, 'save_config') and args.save_config:
            save_config(config, args.save_config)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        return config
    
    def _create_datasets(self, config, args):
        """Create datasets based on configuration."""
        if config.task_type == TaskType.LLM:
            train_dataset, _ = DatasetFactory.create_llm_dataset(
                vocab_size=config.output_dim,
                seq_length=config.sequence_length
            )
            val_dataset = None
            if not args.no_validation:
                val_dataset, _ = DatasetFactory.create_llm_dataset(
                    vocab_size=config.output_dim,
                    seq_length=config.sequence_length,
                    num_samples=5000
                )
        elif config.task_type == TaskType.VISION:
            train_dataset = DatasetFactory.create_vision_dataset(train=True)
            val_dataset = DatasetFactory.create_vision_dataset(train=False) if not args.no_validation else None
        else:
            train_dataset = DatasetFactory.create_robotics_dataset(
                state_dim=config.input_dim,
                action_dim=config.output_dim,
                seq_length=config.sequence_length
            )
            val_dataset = DatasetFactory.create_robotics_dataset(
                state_dim=config.input_dim,
                action_dim=config.output_dim,
                seq_length=config.sequence_length,
                num_samples=1000
            ) if not args.no_validation else None
        
        return train_dataset, val_dataset
    
    def _create_data_loaders(self, config, args, train_dataset, val_dataset):
        """Create data loaders."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2 if config.device == 'cuda' else 0
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2 if config.device == 'cuda' else 0
            )
        
        return train_loader, val_loader
    
    def _benchmark_batch_size(self, model, config, batch_size, args):
        """Benchmark single batch size."""
        # Create dummy input
        if config.task_type == TaskType.VISION:
            dummy_input = torch.randn(batch_size, 3, 32, 32)
        else:
            dummy_input = torch.randn(batch_size, config.sequence_length, config.input_dim)
        
        # Warmup
        model.eval()
        device = torch.device(config.device)
        model.to(device)
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(args.iterations):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / args.iterations
        throughput = batch_size / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
    
    def _display_model_stats(self, model, config):
        """Display model statistics."""
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats_table = Table(title="Model Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Total Parameters", f"{param_count:,}")
        stats_table.add_row("Trainable Parameters", f"{trainable_params:,}")
        stats_table.add_row("Task Type", config.task_type.value)
        stats_table.add_row("Device", config.device)
        
        console.print(stats_table)
    
    def _handle_inference(self, args):
        """Handle inference with rich progress."""
        self.logger.header("Model Inference", f"Model: {Path(args.model_path).name}")
        
        # Load model
        with Status("[bold green]Loading model...", spinner="dots"):
            model, config = load_model(args.model_path, None)
        
        # Prepare input data
        input_data = self._prepare_input_data(args, config)
        
        # Run inference
        with Status("[bold green]Running inference...", spinner="dots"):
            start_time = time.time()
            predictions = inference_example(model, config, input_data)
            inference_time = time.time() - start_time
        
        # Display results
        self._display_inference_results(input_data, predictions, inference_time, args)
        
        # Save results if requested
        if args.output_file:
            self._save_inference_results(predictions, args.output_file)
    
    def _prepare_input_data(self, args, config):
        """Prepare input data for inference."""
        if args.input_file:
            self.logger.info(f"Loading input from {args.input_file}")
            if args.input_file.endswith('.npy'):
                input_data = np.load(args.input_file)
            elif args.input_file.endswith('.pt'):
                input_data = torch.load(args.input_file)
            elif args.input_file.endswith('.json'):
                with open(args.input_file, 'r') as f:
                    input_data = np.array(json.load(f))
            else:
                raise ValueError(f"Unsupported input file format: {args.input_file}")
        elif args.input_shape:
            shape = tuple(map(int, args.input_shape.split(',')))
            self.logger.info(f"Generating random input with shape {shape}")
            input_data = np.random.randn(*shape)
        else:
            # Use default shape based on task type
            if config.task_type == TaskType.VISION:
                input_data = np.random.randn(3, 32, 32)
            else:
                input_data = np.random.randn(config.sequence_length, config.input_dim)
            self.logger.info(f"Using default random input with shape {input_data.shape}")
        
        return input_data
    
    def _display_inference_results(self, input_data, predictions, inference_time, args):
        """Display inference results in a table."""
        results_table = Table(title="Inference Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Input Shape", str(input_data.shape))
        results_table.add_row("Output Shape", str(predictions.shape))
        results_table.add_row("Inference Time", f"{inference_time*1000:.2f} ms")
        
        if args.verbose and predictions.size < 20:  # Only show predictions if small
            results_table.add_row("Predictions", str(predictions.numpy() if hasattr(predictions, 'numpy') else predictions))
        
        console.print(results_table)
    
    def _save_inference_results(self, predictions, output_file):
        """Save inference results."""
        if output_file.endswith('.npy'):
            np.save(output_file, predictions)
        elif output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(predictions.tolist(), f, indent=2)
        else:
            np.save(output_file, predictions)
        self.logger.success(f"Results saved to {output_file}")
    
    def _handle_export(self, args):
        """Handle export with progress."""
        self.logger.header("Model Export", f"Format: {args.format.upper()}")
        
        # Load model
        with Status("[bold green]Loading model...", spinner="dots"):
            model, config = load_model(args.model_path, None)
        
        # Export model
        with Status(f"[bold green]Exporting to {args.format}...", spinner="dots"):
            if args.format == 'onnx':
                export_onnx(model, config, args.output_path)
            elif args.format == 'torchscript':
                self._export_torchscript(model, config, args.output_path)
        
        self.logger.success(f"Model exported successfully to {args.output_path}")
    
    def _export_torchscript(self, model, config, output_path):
        """Export model to TorchScript format."""
        model.eval()
        
        if config.task_type == TaskType.VISION:
            dummy_input = torch.randn(1, 3, 32, 32)
        else:
            dummy_input = torch.randn(1, config.sequence_length, config.input_dim)
        
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
    
    def _handle_config(self, args):
        """Handle configuration with interactive editor."""
        self.logger.header("Configuration Management", f"Task: {args.task.upper()}")
        
        # Create base configuration
        if args.task == 'llm':
            config = create_llm_config()
        elif args.task == 'vision':
            config = create_vision_config()
        else:
            config = create_robotics_config()
        
        # Apply modifications if provided
        if args.modify:
            modifications = json.loads(args.modify)
            config_dict = config.to_dict()
            config_dict.update(modifications)
            config = ModelConfig.from_dict(config_dict)
            self.logger.info(f"Applied modifications: {modifications}")
        
        # Interactive editor
        if args.interactive:
            config = self._interactive_config_editor(config)
        
        # Display final configuration
        self._display_config_table(config)
        
        # Save configuration
        with open(args.save_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        self.logger.success(f"Configuration saved to {args.save_path}")
    
    def _interactive_config_editor(self, config):
        """Interactive configuration editor using Rich prompts."""
        console.print("\n[bold blue]Interactive Configuration Editor[/bold blue]")
        console.print("[dim]Press Enter to keep current value, or enter new value[/dim]\n")
        
        config_dict = config.to_dict()
        
        for key, value in config_dict.items():
            if key == 'task_type':
                continue  # Don't allow changing task type
            
            if isinstance(value, bool):
                new_value = Confirm.ask(f"{key}", default=value)
            elif isinstance(value, int):
                new_value = IntPrompt.ask(f"{key}", default=value)
            elif isinstance(value, float):
                new_value = FloatPrompt.ask(f"{key}", default=value)
            else:
                new_value = Prompt.ask(f"{key}", default=str(value))
            
            config_dict[key] = new_value
        
        return ModelConfig.from_dict(config_dict)
    
    def _display_model_info_detailed(self, model_path, config_only=False):
        """Display detailed model information."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = ModelConfig.from_dict(checkpoint['config'])
            
            # Configuration table
            config_table = Table(title=f"Model Configuration - {Path(model_path).name}")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_dict = config.to_dict()
            for key, value in config_dict.items():
                if key == 'task_type':
                    value = value.value if hasattr(value, 'value') else value
                config_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(config_table)
            
            if not config_only:
                # Training information
                training_table = Table(title="Training Information")
                training_table.add_column("Metric", style="cyan")
                training_table.add_column("Value", style="white")
                
                if 'train_losses' in checkpoint:
                    training_table.add_row("Training Epochs", str(len(checkpoint['train_losses'])))
                    training_table.add_row("Final Train Loss", f"{checkpoint['train_losses'][-1]:.4f}")
                if 'val_losses' in checkpoint and checkpoint['val_losses']:
                    training_table.add_row("Final Val Loss", f"{checkpoint['val_losses'][-1]:.4f}")
                if 'best_val_loss' in checkpoint:
                    training_table.add_row("Best Val Loss", f"{checkpoint['best_val_loss']:.4f}")
                
                # Model statistics
                model = LiquidSpikingNetwork(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                training_table.add_row("Total Parameters", f"{total_params:,}")
                training_table.add_row("Trainable Parameters", f"{trainable_params:,}")
                
                console.print(training_table)
                
        except Exception as e:
            self.logger.error(f"Failed to load model info: {str(e)}")

def main():
    """Main entry point for the CLI application."""
    cli = LiquidSpikingCLI()
    cli.run()

if __name__ == "__main__":
    main()
