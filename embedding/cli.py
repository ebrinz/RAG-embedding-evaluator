"""Command-line interface for embedding evaluations."""
import typer
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from loguru import logger
import json
from typing import Optional

from .evaluator import EmbeddingEvaluator
from .data_loader import DataLoader
from .logger_config import setup_logger, get_progress

# Setup
logger = setup_logger()
console = Console()
app = typer.Typer()

def load_config(config_path: Path) -> dict:
    """Load evaluation configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Found {len(config.get('models', []))} models in configuration")
    return config

@app.command()
def download(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to evaluation config YAML"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reload of dataset")
):
    """Load and verify the evaluation dataset."""
    console.print(Panel.fit(
        "[bold blue]Loading Dataset[/bold blue]",
        border_style="blue"
    ))
    
    data_loader = DataLoader(config_path)
    
    with get_progress() as progress:
        task = progress.add_task(
            "[cyan]Loading and preparing dataset...",
            total=None
        )
        try:
            df = data_loader.load_dataset(force_reload=force)  # Changed parameter name here
            progress.update(task, completed=True)
            
            console.print(Panel.fit(
                f"[bold green]Dataset loaded successfully![/bold green]\n"
                f"Total items: {len(df)}",
                border_style="green"
            ))
        except Exception as e:
            console.print(Panel.fit(
                f"[bold red]Error loading dataset:[/bold red]\n{str(e)}",
                border_style="red"
            ))
            raise typer.Exit(1)

@app.command()
def prepare(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to evaluation config YAML"),
    force: bool = typer.Option(False, "--force", "-f", help="Force regeneration of test data")
):
    """Prepare test data for evaluation."""
    # First ensure dataset is loaded
    download(config_path, force=force)
    
    console.print(Panel.fit(
        "[bold blue]Preparing Test Data[/bold blue]",
        border_style="blue"
    ))
    
    data_loader = DataLoader(config_path)
    
    with get_progress() as progress:
        test_data_path = data_loader.prepare_test_data(progress)
    
    console.print(Panel.fit(
        f"[bold green]Test data prepared successfully![/bold green]\n"
        f"Location: {test_data_path}",
        border_style="green"
    ))
    
    return test_data_path

@app.command()
def evaluate(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to evaluation config YAML"),
    data_path: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to test data JSON"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Path to save results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Run embedding model evaluation."""
    # First ensure dataset is loaded and prepared
    if data_path is None:
        data_path = prepare(config_path)
    
    console.print(Panel.fit(
        "[bold green]Starting Embedding Model Evaluation[/bold green]",
        border_style="blue"
    ))
    
    # Load configuration
    config = load_config(config_path)
    display_config_summary(config)
    
    # Initialize evaluator
    evaluator = EmbeddingEvaluator(config_path)
    
    with get_progress() as progress:
        task = progress.add_task(
            "[cyan]Running evaluation...",
            total=None
        )
        
        results_df = evaluator.run_evaluation()
        progress.update(task, completed=True)
    
    # Save and display results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    display_results(results_df)
    
    console.print(Panel.fit(
        "[bold green]Evaluation Complete![/bold green]\n"
        f"Results saved to: {output_path}",
        border_style="green"
    ))

def display_config_summary(config: dict):
    """Display a summary of the evaluation configuration."""
    console.print("\n[bold blue]Evaluation Configuration Summary:[/bold blue]")
    
    # Models table
    model_table = Table(title="Models to Evaluate")
    model_table.add_column("Model Name")
    model_table.add_column("Description")
    
    for model in config['models']:
        model_table.add_row(
            model['name'],
            model.get('description', 'No description provided')
        )
    
    console.print(model_table)
    
    # Other settings
    settings_table = Table(title="Evaluation Settings")
    settings_table.add_column("Setting")
    settings_table.add_column("Value")
    
    settings_table.add_row("Batch Size", str(config.get('batch_size', 32)))
    settings_table.add_row("Device", config.get('device', 'cpu'))
    settings_table.add_row("Max Length", str(config.get('max_length', 512)))
    
    console.print(settings_table)

def display_results(results_df):
    """Display evaluation results in a formatted table."""
    console.print("\n[bold green]Evaluation Results:[/bold green]")
    table = Table(title="Model Performance Comparison")
    
    # Add columns
    table.add_column("Model", style="cyan")
    for col in results_df.columns:
        if col != 'model' and not col.startswith('description'):
            table.add_column(col, justify="right")
    
    # Add rows
    for _, row in results_df.iterrows():
        values = [row['model']] + [
            f"{row[col]:.3f}" if col != 'model' and not col.startswith('description')
            else row[col] for col in results_df.columns
            if col != 'description'
        ]
        table.add_row(*values)
    
    console.print(table)

if __name__ == "__main__":
    app()