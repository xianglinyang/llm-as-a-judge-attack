#!/usr/bin/env python3
"""
Transfer Effectiveness Heatmap Visualization

This script creates a heatmap visualization of transfer effectiveness between different judge models.
The heatmap shows how well attacks optimized for one judge transfer to other judges.

Structure:
- Rows: Source Judge (the judge the attack was optimized on)
- Columns: Target Judge (the judge the optimized response is being tested against)
- Cell Color: Transfer Effectiveness (normalized score)
- Cell Annotation: Transfer ASR (raw success rate)

Color Scheme:
- Hot Color (bright green/blue): High effectiveness (near 1.0 or 100%)
- Cold Color (white/pale yellow): Low effectiveness (near 0.0)
- Negative Color (red): Negative effectiveness (harmful)

"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass

@dataclass
class TransferData:
    """Container for transfer analysis data."""
    source_judge: str
    target_judge: str
    dataset: str
    source_asr: float
    target_asr: float
    transfer_asr: float
    transfer_effectiveness: float
    num_questions: int

class TransferHeatmapVisualizer:
    """Main class for creating transfer effectiveness heatmaps."""
    
    def __init__(self, reports_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            reports_dir: Directory containing transfer analysis reports
        """
        self.reports_dir = Path(reports_dir)
        self.transfer_data: List[TransferData] = []
        
    def parse_report_file(self, file_path: Path) -> List[TransferData]:
        """
        Parse a single transfer analysis report file that may contain multiple datasets.
        
        Args:
            file_path: Path to the report file
            
        Returns:
            List of TransferData objects (one per dataset in the file)
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract source and target judge from filename
            filename = file_path.stem
            match = re.match(r'(.+)_to_(.+)', filename)
            if not match:
                print(f"Warning: Could not parse judge names from filename {filename}")
                return []
            
            source_judge = match.group(1)
            target_judge = match.group(2)
            
            # Split content by dataset sections
            # Look for pattern like "## source_judge → target_judge" followed by dataset info
            section_pattern = rf'## {re.escape(source_judge)} → {re.escape(target_judge)}.*?(?=## {re.escape(source_judge)} → {re.escape(target_judge)}|$)'
            sections = re.findall(section_pattern, content, re.DOTALL)
            
            if not sections:
                # If no sections found, treat entire content as one section
                sections = [content]
            
            transfer_data_list = []
            
            for section in sections:
                # Extract dataset name
                dataset_match = re.search(r'\*\*Dataset\*\*:\s+(\w+)', section)
                if not dataset_match:
                    print(f"Warning: Could not find dataset in section of {file_path}")
                    continue
                
                dataset = dataset_match.group(1)
                
                # Extract data using regex patterns
                patterns = {
                    'source_asr': r'\*\*Source ASR\*\*:\s+(\d+\.?\d*)%',
                    'target_asr': r'\*\*Target ASR\*\*:\s+(\d+\.?\d*)%',
                    'transfer_asr': r'\*\*Transfer ASR\*\*:\s+(\d+\.?\d*)%',
                    'transfer_effectiveness': r'\*\*Transfer Effectiveness\*\*:\s+(-?\d+\.?\d*)',
                    'num_questions': r'\*\*Questions Analyzed\*\*:\s+(\d+)'
                }
                
                extracted_data = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, section)
                    if match:
                        extracted_data[key] = float(match.group(1))
                    else:
                        print(f"Warning: Could not find {key} in {dataset} section of {file_path}")
                        break
                else:
                    # All patterns found successfully
                    transfer_data = TransferData(
                        source_judge=source_judge,
                        target_judge=target_judge,
                        dataset=dataset,
                        source_asr=extracted_data['source_asr'],
                        target_asr=extracted_data['target_asr'],
                        transfer_asr=extracted_data['transfer_asr'],
                        transfer_effectiveness=extracted_data['transfer_effectiveness'],
                        num_questions=int(extracted_data['num_questions'])
                    )
                    transfer_data_list.append(transfer_data)
            
            return transfer_data_list
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def load_transfer_data(self) -> None:
        """Load all transfer data from report files."""
        transfer_files = list(self.reports_dir.glob("*_to_*.md"))
        
        if not transfer_files:
            print(f"Warning: No transfer report files found in {self.reports_dir}")
            return
            
        print(f"Found {len(transfer_files)} transfer report files")
        
        for file_path in transfer_files:
            transfer_data_list = self.parse_report_file(file_path)
            for transfer_data in transfer_data_list:
                self.transfer_data.append(transfer_data)
                print(f"Loaded: {transfer_data.source_judge} → {transfer_data.target_judge} ({transfer_data.dataset})")
                
        print(f"Successfully loaded {len(self.transfer_data)} transfer results")
    
    def create_transfer_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create matrices for transfer effectiveness, transfer ASR, and number of questions.
        
        Returns:
            Tuple of (effectiveness_matrix, asr_matrix, questions_matrix)
        """
        if not self.transfer_data:
            raise ValueError("No transfer data loaded. Call load_transfer_data() first.")
        
        # Get all unique judges
        all_judges = set()
        for data in self.transfer_data:
            all_judges.add(data.source_judge)
            all_judges.add(data.target_judge)
        all_judges = sorted(list(all_judges))
        
        # Initialize matrices with NaN
        effectiveness_matrix = pd.DataFrame(index=all_judges, columns=all_judges, dtype=float)
        asr_matrix = pd.DataFrame(index=all_judges, columns=all_judges, dtype=float)
        questions_matrix = pd.DataFrame(index=all_judges, columns=all_judges, dtype=int)
        
        # Fill matrices with data
        for data in self.transfer_data:
            effectiveness_matrix.loc[data.source_judge, data.target_judge] = data.transfer_effectiveness
            asr_matrix.loc[data.source_judge, data.target_judge] = data.transfer_asr
            questions_matrix.loc[data.source_judge, data.target_judge] = data.num_questions
        
        # Fill diagonal with perfect transfer (effectiveness = 1.0, ASR = source ASR)
        for judge in all_judges:
            effectiveness_matrix.loc[judge, judge] = 1.0
            # Find source ASR for this judge
            source_asr = None
            for data in self.transfer_data:
                if data.source_judge == judge:
                    source_asr = data.source_asr
                    break
            if source_asr is not None:
                asr_matrix.loc[judge, judge] = source_asr
            
        return effectiveness_matrix, asr_matrix, questions_matrix
    
    def create_effectiveness_heatmap(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Create and display the transfer effectiveness heatmap.
        
        Args:
            save_path: Path to save the heatmap image (optional)
            figsize: Figure size as (width, height)
        """
        effectiveness_matrix, asr_matrix, questions_matrix = self.create_transfer_matrices()
        
        # Create figure and axis
        plt.figure(figsize=figsize)
        
        # Create custom colormap
        # Red for negative, white for zero, yellow to green to blue for positive
        colors = ['#FF4444', '#FFFFFF', '#FFFF99', '#90EE90', '#87CEEB', '#4169E1']
        n_bins = 100
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('transfer', colors, N=n_bins)
        
        # Determine color range
        min_val = effectiveness_matrix.min().min()
        max_val = effectiveness_matrix.max().max()
        
        # Make sure the range is symmetric around 0 for proper color mapping
        abs_max = max(abs(min_val), abs(max_val))
        vmin, vmax = -abs_max, abs_max
        
        # Create heatmap
        mask = effectiveness_matrix.isna()
        
        ax = sns.heatmap(
            effectiveness_matrix.astype(float),
            annot=False,  # We'll add custom annotations
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Transfer Effectiveness'},
            mask=mask
        )
        
        # Add custom annotations (Transfer ASR values)
        for i in range(len(effectiveness_matrix.index)):
            for j in range(len(effectiveness_matrix.columns)):
                if not mask.iloc[i, j]:
                    effectiveness = effectiveness_matrix.iloc[i, j]
                    asr = asr_matrix.iloc[i, j]
                    
                    if not pd.isna(effectiveness) and not pd.isna(asr):
                        # Format text based on values
                        text_color = 'white' if abs(effectiveness) > 0.5 else 'black'
                        
                        # Show both effectiveness and ASR
                        text = f'{effectiveness:.2f}\n({asr:.1f}%)'
                        ax.text(j + 0.5, i + 0.5, text, 
                               ha='center', va='center', color=text_color, fontsize=9)
        
        # Customize the plot
        plt.title('Transfer Effectiveness Heatmap\n(Values: Effectiveness, ASR)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Target Judge', fontsize=14, fontweight='bold')
        plt.ylabel('Source Judge', fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report with transfer matrix.
        
        Returns:
            Formatted markdown report
        """
        if not self.transfer_data:
            return "No transfer data available for report generation."
        
        effectiveness_matrix, asr_matrix, questions_matrix = self.create_transfer_matrices()
        
        report_lines = []
        report_lines.append("# Transfer Effectiveness Analysis Report")
        report_lines.append("")
        report_lines.append("## Overview")
        report_lines.append("")
        report_lines.append("This report presents a comprehensive analysis of attack transfer effectiveness ")
        report_lines.append("between different judge models. The analysis shows how well attacks optimized ")
        report_lines.append("for one judge model transfer to other judge models.")
        report_lines.append("")
        
        # Summary statistics
        valid_effectiveness = effectiveness_matrix.values[~pd.isna(effectiveness_matrix.values)]
        valid_effectiveness = valid_effectiveness[valid_effectiveness != 1.0]  # Exclude diagonal
        
        if len(valid_effectiveness) > 0:
            report_lines.append("## Summary Statistics")
            report_lines.append("")
            report_lines.append(f"- **Total Transfer Pairs**: {len(self.transfer_data)}")
            report_lines.append(f"- **Average Transfer Effectiveness**: {np.mean(valid_effectiveness):.3f}")
            report_lines.append(f"- **Best Transfer Effectiveness**: {np.max(valid_effectiveness):.3f}")
            report_lines.append(f"- **Worst Transfer Effectiveness**: {np.min(valid_effectiveness):.3f}")
            report_lines.append(f"- **Standard Deviation**: {np.std(valid_effectiveness):.3f}")
            report_lines.append("")
        
        # Transfer Effectiveness Matrix
        report_lines.append("## Transfer Effectiveness Matrix")
        report_lines.append("")
        report_lines.append("| Source \\ Target | " + " | ".join(effectiveness_matrix.columns) + " |")
        report_lines.append("| " + " | ".join(["---"] * (len(effectiveness_matrix.columns) + 1)) + " |")
        
        for idx in effectiveness_matrix.index:
            row_data = [idx]
            for col in effectiveness_matrix.columns:
                val = effectiveness_matrix.loc[idx, col]
                if pd.isna(val):
                    row_data.append("N/A")
                elif idx == col:
                    row_data.append("1.00")  # Diagonal (perfect self-transfer)
                else:
                    row_data.append(f"{val:.3f}")
            report_lines.append("| " + " | ".join(row_data) + " |")
        
        report_lines.append("")
        
        # Transfer ASR Matrix
        report_lines.append("## Transfer ASR Matrix (%)")
        report_lines.append("")
        report_lines.append("| Source \\ Target | " + " | ".join(asr_matrix.columns) + " |")
        report_lines.append("| " + " | ".join(["---"] * (len(asr_matrix.columns) + 1)) + " |")
        
        for idx in asr_matrix.index:
            row_data = [idx]
            for col in asr_matrix.columns:
                val = asr_matrix.loc[idx, col]
                if pd.isna(val):
                    row_data.append("N/A")
                else:
                    row_data.append(f"{val:.1f}")
            report_lines.append("| " + " | ".join(row_data) + " |")
        
        report_lines.append("")
        
        # Individual Transfer Results
        report_lines.append("## Individual Transfer Results")
        report_lines.append("")
        
        for data in sorted(self.transfer_data, key=lambda x: (x.source_judge, x.target_judge)):
            report_lines.append(f"### {data.source_judge} → {data.target_judge}")
            report_lines.append("")
            report_lines.append(f"- **Transfer Effectiveness**: {data.transfer_effectiveness:.3f}")
            report_lines.append(f"- **Transfer ASR**: {data.transfer_asr:.1f}%")
            report_lines.append(f"- **Source ASR**: {data.source_asr:.1f}%")
            report_lines.append(f"- **Target ASR**: {data.target_asr:.1f}%")
            report_lines.append(f"- **Questions Analyzed**: {data.num_questions}")
            report_lines.append("")
        
        # Color scheme explanation
        report_lines.append("## Heatmap Color Scheme")
        report_lines.append("")
        report_lines.append("- **Blue/Green**: High positive transfer effectiveness (≥ 0.5)")
        report_lines.append("- **Yellow**: Moderate positive transfer effectiveness (0.0 to 0.5)")
        report_lines.append("- **White**: No transfer effectiveness (≈ 0.0)")
        report_lines.append("- **Red**: Negative transfer effectiveness (< 0.0)")
        report_lines.append("- **Diagonal**: Perfect self-transfer (1.0)")
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def get_dataset_summary(self, dataset_name: str) -> Dict:
        """
        Collect summary statistics for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to summarize (e.g., 'AlpacaEval', 'ArenaHard')
            
        Returns:
            Dictionary containing summary statistics for the dataset
        """
        if not self.transfer_data:
            raise ValueError("No transfer data loaded. Call load_transfer_data() first.")
        
        # Filter data for the specified dataset
        dataset_data = [data for data in self.transfer_data if data.dataset == dataset_name]
        
        if not dataset_data:
            return {
                'dataset': dataset_name,
                'error': f'No data found for dataset: {dataset_name}',
                'available_datasets': list(set(data.dataset for data in self.transfer_data))
            }
        
        # Extract metrics
        transfer_asrs = [data.transfer_asr for data in dataset_data]
        transfer_effectivenesses = [data.transfer_effectiveness for data in dataset_data]
        source_asrs = [data.source_asr for data in dataset_data]
        target_asrs = [data.target_asr for data in dataset_data]
        
        # Get unique judges
        source_judges = set(data.source_judge for data in dataset_data)
        target_judges = set(data.target_judge for data in dataset_data)
        all_judges = source_judges.union(target_judges)
        
        # Calculate summary statistics
        summary = {
            'dataset': dataset_name,
            'total_transfer_pairs': len(dataset_data),
            'unique_judges': len(all_judges),
            'judge_list': sorted(list(all_judges)),
            'transfer_asr': {
                'mean': np.mean(transfer_asrs),
                'std': np.std(transfer_asrs),
                'min': np.min(transfer_asrs),
                'max': np.max(transfer_asrs),
                'median': np.median(transfer_asrs)
            },
            'transfer_effectiveness': {
                'mean': np.mean(transfer_effectivenesses),
                'std': np.std(transfer_effectivenesses),
                'min': np.min(transfer_effectivenesses),
                'max': np.max(transfer_effectivenesses),
                'median': np.median(transfer_effectivenesses)
            },
            'source_asr': {
                'mean': np.mean(source_asrs),
                'std': np.std(source_asrs),
                'min': np.min(source_asrs),
                'max': np.max(source_asrs),
                'median': np.median(source_asrs)
            },
            'target_asr': {
                'mean': np.mean(target_asrs),
                'std': np.std(target_asrs),
                'min': np.min(target_asrs),
                'max': np.max(target_asrs),
                'median': np.median(target_asrs)
            }
        }
        
        # Add best and worst performing pairs
        best_transfer_idx = np.argmax(transfer_effectivenesses)
        worst_transfer_idx = np.argmin(transfer_effectivenesses)
        
        summary['best_transfer'] = {
            'source_judge': dataset_data[best_transfer_idx].source_judge,
            'target_judge': dataset_data[best_transfer_idx].target_judge,
            'transfer_asr': dataset_data[best_transfer_idx].transfer_asr,
            'transfer_effectiveness': dataset_data[best_transfer_idx].transfer_effectiveness
        }
        
        summary['worst_transfer'] = {
            'source_judge': dataset_data[worst_transfer_idx].source_judge,
            'target_judge': dataset_data[worst_transfer_idx].target_judge,
            'transfer_asr': dataset_data[worst_transfer_idx].transfer_asr,
            'transfer_effectiveness': dataset_data[worst_transfer_idx].transfer_effectiveness
        }
        
        return summary
    
    def create_transfer_matrices_for_dataset(self, dataset_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create matrices for a specific dataset or all datasets combined.
        
        Args:
            dataset_name: Name of the dataset to create matrices for. If None, use all datasets.
            
        Returns:
            Tuple of (effectiveness_matrix, asr_matrix, questions_matrix)
        """
        if not self.transfer_data:
            raise ValueError("No transfer data loaded. Call load_transfer_data() first.")
        
        # Filter data by dataset if specified
        if dataset_name:
            filtered_data = [data for data in self.transfer_data if data.dataset == dataset_name]
            if not filtered_data:
                available_datasets = list(set(data.dataset for data in self.transfer_data))
                raise ValueError(f"No data found for dataset: {dataset_name}. Available datasets: {available_datasets}")
        else:
            filtered_data = self.transfer_data
        
        # Get all unique judges
        all_judges = set()
        for data in filtered_data:
            all_judges.add(data.source_judge)
            all_judges.add(data.target_judge)
        all_judges = sorted(list(all_judges))
        
        # Initialize matrices with NaN
        effectiveness_matrix = pd.DataFrame(index=all_judges, columns=all_judges, dtype=float)
        asr_matrix = pd.DataFrame(index=all_judges, columns=all_judges, dtype=float)
        questions_matrix = pd.DataFrame(index=all_judges, columns=all_judges, dtype=int)
        
        # Fill matrices with data
        for data in filtered_data:
            effectiveness_matrix.loc[data.source_judge, data.target_judge] = data.transfer_effectiveness
            asr_matrix.loc[data.source_judge, data.target_judge] = data.transfer_asr
            questions_matrix.loc[data.source_judge, data.target_judge] = data.num_questions
        
        # Fill diagonal with perfect transfer (effectiveness = 1.0, ASR = source ASR)
        for judge in all_judges:
            effectiveness_matrix.loc[judge, judge] = 1.0
            # Find source ASR for this judge
            source_asr = None
            for data in filtered_data:
                if data.source_judge == judge:
                    source_asr = data.source_asr
                    break
            if source_asr is not None:
                asr_matrix.loc[judge, judge] = source_asr
            
        return effectiveness_matrix, asr_matrix, questions_matrix
    
    def create_asr_heatmap(self, dataset_name: Optional[str] = None, save_path: Optional[str] = None, 
                          figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Create and display a transfer ASR heatmap for a specific dataset or all datasets.
        
        Args:
            dataset_name: Name of the dataset to visualize. If None, use all datasets.
            save_path: Path to save the heatmap image (optional)
            figsize: Figure size as (width, height)
        """
        effectiveness_matrix, asr_matrix, questions_matrix = self.create_transfer_matrices_for_dataset(dataset_name)

        # set the diagonal to 100
        np.fill_diagonal(asr_matrix.values, 100)
        
        # Create figure and axis
        plt.figure(figsize=figsize)
        
        # Create custom colormap for ASR (0-100%)
        colors = ['#FFFFFF', '#FFF2CC', '#FFE699', '#FFD966', '#FFCC33', '#FF9900', '#FF6600', '#CC3300']
        n_bins = 100
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('asr', colors, N=n_bins)
        
        # Create heatmap
        mask = asr_matrix.isna()
        
        ax = sns.heatmap(
            asr_matrix.astype(float),
            annot=True,
            fmt='.1f',
            cmap=cmap,
            vmin=0,
            vmax=100,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Transfer ASR (%)'},
            mask=mask
        )
        
        # Customize the plot
        title = f'Transfer ASR Heatmap'
        if dataset_name:
            title += f' - {dataset_name} Dataset'
        # else:
            # title += ' - All Datasets'

        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Target Judge', fontsize=14, fontweight='bold')
        plt.ylabel('Source Judge', fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ASR heatmap saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_heatmaps(self, save_dir: Optional[str] = None, figsize: Tuple[int, int] = (15, 6)) -> None:
        """
        Create side-by-side comparison heatmaps for all available datasets.
        
        Args:
            save_dir: Directory to save the heatmap images (optional)
            figsize: Figure size as (width, height)
        """
        if not self.transfer_data:
            raise ValueError("No transfer data loaded. Call load_transfer_data() first.")
        
        # Get all available datasets
        available_datasets = sorted(list(set(data.dataset for data in self.transfer_data)))
        
        if len(available_datasets) == 0:
            print("No datasets found in the transfer data.")
            return
        
        # Create subplots for each dataset
        fig, axes = plt.subplots(1, len(available_datasets), figsize=(figsize[0] * len(available_datasets), figsize[1]))
        
        # Handle case with single dataset
        if len(available_datasets) == 1:
            axes = [axes]
        
        for i, dataset in enumerate(available_datasets):
            effectiveness_matrix, asr_matrix, questions_matrix = self.create_transfer_matrices_for_dataset(dataset)
            # set the diagonal to 100
            np.fill_diagonal(asr_matrix.values, 100)
            
            # Create custom colormap for ASR (0-100%)
            colors = ['#FFFFFF', '#FFF2CC', '#FFE699', '#FFD966', '#FFCC33', '#FF9900', '#FF6600', '#CC3300']
            n_bins = 100
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('asr', colors, N=n_bins)
            
            # Create heatmap
            mask = asr_matrix.isna()
            
            sns.heatmap(
                asr_matrix.astype(float),
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=0,
                vmax=100,
                square=True,
                linewidths=0.5,
                cbar=True,
                mask=mask,
                ax=axes[i]
            )
            
            # Customize subplot
            axes[i].set_title(f'{dataset} Dataset', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Target Judge', fontsize=12, fontweight='bold')
            if i == 0:
                axes[i].set_ylabel('Source Judge', fontsize=12, fontweight='bold')
            else:
                axes[i].set_ylabel('')
            
            # Rotate labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
        
        # Main title
        fig.suptitle('Transfer ASR Comparison Across Datasets', fontsize=18, fontweight='bold', y=1.02)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save if directory provided
        if save_dir:
            save_path = Path(save_dir) / "transfer_asr_comparison.pdf"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison heatmaps saved to: {save_path}")
        
        plt.show()

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate Transfer Effectiveness Heatmap")
    parser.add_argument("--reports_dir", type=str, 
                       default="./reports",
                       help="Directory containing transfer analysis reports")
    parser.add_argument("--output_dir", type=str,
                       default="./reports",
                       help="Directory to save output files")
    parser.add_argument("--figsize", nargs=2, type=int, default=[12, 10],
                       help="Figure size as width height")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TransferHeatmapVisualizer(args.reports_dir)
    
    # Load data
    print("Loading transfer analysis data...")
    visualizer.load_transfer_data()
    
    if not visualizer.transfer_data:
        print("No transfer data found. Please ensure transfer analysis reports exist.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison heatmaps for all datasets
    save_path = output_dir / "comparison_heatmaps.pdf"
    visualizer.create_comparison_heatmaps(save_dir=output_dir)

    # Create combined ASR heatmap
    save_path = output_dir / "combined_asr.pdf"
    visualizer.create_asr_heatmap(dataset_name=None, save_path=save_path)

    # Create combined ASR heatmap
    save_path = output_dir / "alpaca_eval_asr.pdf"
    visualizer.create_asr_heatmap(dataset_name="AlpacaEval", save_path=save_path)

    # Create combined ASR heatmap
    save_path = output_dir / "arena_hard_asr.pdf"
    visualizer.create_asr_heatmap(dataset_name="ArenaHard", save_path=save_path)

    # Generate comprehensive report
    print("Generating comprehensive transfer analysis report...")
    report = visualizer.generate_summary_report()
    
    # Save report
    report_path = output_dir / "transfer_summary.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to: {report_path}")
    print("\n✅ Transfer effectiveness analysis completed successfully!")

if __name__ == "__main__":
    main()
