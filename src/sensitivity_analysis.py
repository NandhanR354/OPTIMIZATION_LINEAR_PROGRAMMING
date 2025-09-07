"""
Sensitivity Analysis for Optimization Model
Author: CODTECH Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from optimization_model import ProductionOptimizer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class SensitivityAnalyzer:
    """Perform sensitivity analysis on optimization parameters"""

    def __init__(self):
        self.logger = setup_logging()
        self.base_optimizer = ProductionOptimizer()
        self.sensitivity_results = {}

    def analyze_resource_sensitivity(self, resource_name: str, variation_range: tuple = (0.5, 1.5), steps: int = 10):
        """Analyze sensitivity to resource availability changes"""
        self.logger.info(f"Analyzing sensitivity for {resource_name}")

        base_availability = self.base_optimizer.resource_availability[resource_name]
        multipliers = np.linspace(variation_range[0], variation_range[1], steps)

        results = []
        for multiplier in multipliers:
            # Create modified optimizer
            optimizer = ProductionOptimizer()
            optimizer.resource_availability[resource_name] = base_availability * multiplier

            # Solve
            optimizer.formulate_problem()
            success = optimizer.solve_problem()

            if success:
                results.append({
                    'multiplier': multiplier,
                    'resource_level': base_availability * multiplier,
                    'optimal_profit': optimizer.results['objective_value'],
                    'production_plan': optimizer.results['production_plan'].copy()
                })

        self.sensitivity_results[f'{resource_name}_sensitivity'] = results
        return results

    def analyze_profit_sensitivity(self, product_name: str, variation_range: tuple = (0.5, 1.5), steps: int = 10):
        """Analyze sensitivity to profit margin changes"""
        self.logger.info(f"Analyzing profit sensitivity for {product_name}")

        base_profit = self.base_optimizer.profit_per_unit[product_name]
        multipliers = np.linspace(variation_range[0], variation_range[1], steps)

        results = []
        for multiplier in multipliers:
            # Create modified optimizer
            optimizer = ProductionOptimizer()
            optimizer.profit_per_unit[product_name] = base_profit * multiplier

            # Solve
            optimizer.formulate_problem()
            success = optimizer.solve_problem()

            if success:
                results.append({
                    'multiplier': multiplier,
                    'profit_level': base_profit * multiplier,
                    'optimal_profit': optimizer.results['objective_value'],
                    'production_plan': optimizer.results['production_plan'].copy()
                })

        self.sensitivity_results[f'{product_name}_profit_sensitivity'] = results
        return results

    def generate_sensitivity_plots(self):
        """Generate sensitivity analysis plots"""
        outputs_dir = Path('outputs')
        outputs_dir.mkdir(exist_ok=True)

        # Plot resource sensitivity
        for analysis_name, results in self.sensitivity_results.items():
            if 'sensitivity' in analysis_name:
                plt.figure(figsize=(10, 6))

                multipliers = [r['multiplier'] for r in results]
                profits = [r['optimal_profit'] for r in results]

                plt.plot(multipliers, profits, 'b-o', linewidth=2, markersize=6)
                plt.xlabel('Parameter Multiplier')
                plt.ylabel('Optimal Profit ($)')
                plt.title(f'Sensitivity Analysis: {analysis_name.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(outputs_dir / f'{analysis_name}_plot.png', dpi=300, bbox_inches='tight')
                plt.close()

        self.logger.info("Sensitivity plots generated")

    def run_full_analysis(self):
        """Run comprehensive sensitivity analysis"""
        self.logger.info("Starting comprehensive sensitivity analysis...")

        # Analyze resource sensitivities
        for resource in ['Labor_Hours', 'Raw_Material']:
            self.analyze_resource_sensitivity(resource)

        # Analyze profit sensitivities
        for product in ['Product_A', 'Product_B', 'Product_C']:
            self.analyze_profit_sensitivity(product)

        # Generate plots
        self.generate_sensitivity_plots()

        # Save results
        outputs_dir = Path('outputs')
        results_path = outputs_dir / 'sensitivity_analysis.json'
        with open(results_path, 'w') as f:
            json.dump(self.sensitivity_results, f, indent=2, default=str)

        self.logger.info(f"Sensitivity analysis completed. Results saved to {results_path}")

def main():
    analyzer = SensitivityAnalyzer()
    analyzer.run_full_analysis()
    print("Sensitivity analysis completed!")

if __name__ == "__main__":
    main()
