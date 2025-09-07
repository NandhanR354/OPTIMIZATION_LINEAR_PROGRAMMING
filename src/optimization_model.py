"""
Production Planning Optimization Model using Linear Programming
Author: CODTECH Internship
"""

import pulp
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class ProductionOptimizer:
    """Production planning optimization using linear programming"""

    def __init__(self):
        self.logger = setup_logging()
        self.model = None
        self.variables = {}
        self.constraints = {}
        self.results = {}

        # Problem parameters (can be loaded from file)
        self.products = ['Product_A', 'Product_B', 'Product_C']
        self.resources = ['Labor_Hours', 'Raw_Material']

        # Default parameters
        self.profit_per_unit = {
            'Product_A': 40,
            'Product_B': 30, 
            'Product_C': 50
        }

        self.resource_consumption = {
            ('Product_A', 'Labor_Hours'): 2,
            ('Product_A', 'Raw_Material'): 3,
            ('Product_B', 'Labor_Hours'): 1,
            ('Product_B', 'Raw_Material'): 2,
            ('Product_C', 'Labor_Hours'): 3,
            ('Product_C', 'Raw_Material'): 1
        }

        self.resource_availability = {
            'Labor_Hours': 100,
            'Raw_Material': 80
        }

        self.minimum_demand = {
            'Product_A': 5,
            'Product_B': 10,
            'Product_C': 8
        }

        self.storage_capacity = 50

    def load_parameters(self, file_path: str = None):
        """Load optimization parameters from CSV file"""
        if file_path and Path(file_path).exists():
            self.logger.info(f"Loading parameters from: {file_path}")
            # Implementation for loading from CSV
            # For demo, using default parameters
        else:
            self.logger.info("Using default parameters")
            self._create_sample_data()

    def _create_sample_data(self):
        """Create sample input data file"""
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)

        # Create sample parameters file
        parameters_data = {
            'Product': ['Product_A', 'Product_B', 'Product_C'],
            'Profit_Per_Unit': [40, 30, 50],
            'Labor_Hours_Per_Unit': [2, 1, 3],
            'Raw_Material_Per_Unit': [3, 2, 1],
            'Minimum_Demand': [5, 10, 8]
        }

        df = pd.DataFrame(parameters_data)
        df.to_csv(data_dir / 'input_parameters.csv', index=False)
        self.logger.info("Sample input parameters created")

    def formulate_problem(self):
        """Formulate the linear programming problem"""
        self.logger.info("Formulating optimization problem...")

        # Create the model
        self.model = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

        # Decision variables (production quantities)
        self.variables = {}
        for product in self.products:
            self.variables[product] = pulp.LpVariable(
                f"x_{product}", 
                lowBound=0, 
                cat='Continuous'
            )

        # Objective function: Maximize profit
        profit_terms = [
            self.profit_per_unit[product] * self.variables[product] 
            for product in self.products
        ]
        self.model += pulp.lpSum(profit_terms), "Total_Profit"

        # Constraints
        self.constraints = {}

        # Resource availability constraints
        for resource in self.resources:
            consumption_terms = [
                self.resource_consumption[(product, resource)] * self.variables[product]
                for product in self.products
            ]
            constraint_name = f"{resource}_Availability"
            self.constraints[constraint_name] = (
                pulp.lpSum(consumption_terms) <= self.resource_availability[resource]
            )
            self.model += self.constraints[constraint_name], constraint_name

        # Minimum demand constraints
        for product in self.products:
            constraint_name = f"Min_Demand_{product}"
            self.constraints[constraint_name] = (
                self.variables[product] >= self.minimum_demand[product]
            )
            self.model += self.constraints[constraint_name], constraint_name

        # Storage capacity constraint
        total_production = pulp.lpSum([self.variables[product] for product in self.products])
        self.constraints["Storage_Capacity"] = total_production <= self.storage_capacity
        self.model += self.constraints["Storage_Capacity"], "Storage_Capacity"

        self.logger.info("Problem formulation completed")

    def solve_problem(self):
        """Solve the optimization problem"""
        self.logger.info("Solving optimization problem...")

        # Solve the problem
        solver_status = self.model.solve()

        # Check solution status
        status = pulp.LpStatus[solver_status]
        self.logger.info(f"Solution status: {status}")

        if status == 'Optimal':
            self._extract_solution()
            return True
        else:
            self.logger.error(f"Problem could not be solved optimally. Status: {status}")
            return False

    def _extract_solution(self):
        """Extract and organize solution results"""
        self.logger.info("Extracting solution results...")

        # Extract variable values
        production_plan = {}
        for product in self.products:
            production_plan[product] = self.variables[product].varValue

        # Calculate metrics
        total_profit = pulp.value(self.model.objective)
        total_production = sum(production_plan.values())

        # Resource utilization
        resource_utilization = {}
        for resource in self.resources:
            used = sum(
                self.resource_consumption[(product, resource)] * production_plan[product]
                for product in self.products
            )
            available = self.resource_availability[resource]
            resource_utilization[resource] = {
                'used': used,
                'available': available,
                'utilization_rate': used / available if available > 0 else 0,
                'slack': available - used
            }

        # Shadow prices (dual values)
        shadow_prices = {}
        for name, constraint in self.model.constraints.items():
            if hasattr(constraint, 'pi') and constraint.pi is not None:
                shadow_prices[name] = constraint.pi

        # Organize results
        self.results = {
            'solution_status': 'Optimal',
            'objective_value': total_profit,
            'production_plan': production_plan,
            'total_production': total_production,
            'resource_utilization': resource_utilization,
            'shadow_prices': shadow_prices,
            'problem_parameters': {
                'profit_per_unit': self.profit_per_unit,
                'resource_consumption': self.resource_consumption,
                'resource_availability': self.resource_availability,
                'minimum_demand': self.minimum_demand,
                'storage_capacity': self.storage_capacity
            }
        }

        self.logger.info(f"Optimal profit: ${total_profit:,.2f}")
        self.logger.info("Production plan:")
        for product, quantity in production_plan.items():
            self.logger.info(f"  {product}: {quantity:.2f} units")

    def generate_report(self):
        """Generate comprehensive optimization report"""
        if not self.results:
            self.logger.error("No results to report. Solve the problem first.")
            return

        self.logger.info("Generating optimization report...")

        # Create outputs directory
        outputs_dir = Path('outputs')
        outputs_dir.mkdir(exist_ok=True)

        # Save detailed results as JSON
        results_path = outputs_dir / 'optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Create production plan CSV
        production_df = pd.DataFrame([
            {
                'Product': product,
                'Optimal_Quantity': quantity,
                'Profit_Per_Unit': self.profit_per_unit[product],
                'Total_Profit_Contribution': quantity * self.profit_per_unit[product]
            }
            for product, quantity in self.results['production_plan'].items()
        ])

        production_path = outputs_dir / 'production_plan.csv'
        production_df.to_csv(production_path, index=False)

        # Resource utilization report
        resource_df = pd.DataFrame([
            {
                'Resource': resource,
                'Used': data['used'],
                'Available': data['available'],
                'Utilization_Rate': f"{data['utilization_rate']*100:.1f}%",
                'Slack': data['slack']
            }
            for resource, data in self.results['resource_utilization'].items()
        ])

        resource_path = outputs_dir / 'resource_utilization.csv'
        resource_df.to_csv(resource_path, index=False)

        self.logger.info(f"Reports saved to {outputs_dir}/")

        return {
            'results_json': results_path,
            'production_plan': production_path,
            'resource_utilization': resource_path
        }

    def print_summary(self):
        """Print a summary of the optimization results"""
        if not self.results:
            print("No results available. Solve the problem first.")
            return

        print("\n" + "="*60)
        print("PRODUCTION PLANNING OPTIMIZATION RESULTS")
        print("="*60)

        print(f"\nOptimal Total Profit: ${self.results['objective_value']:,.2f}")
        print(f"Total Production: {self.results['total_production']:.2f} units")

        print("\nOptimal Production Plan:")
        print("-" * 40)
        for product, quantity in self.results['production_plan'].items():
            profit_contribution = quantity * self.profit_per_unit[product]
            print(f"{product:12}: {quantity:8.2f} units (${profit_contribution:8.2f})")

        print("\nResource Utilization:")
        print("-" * 50)
        for resource, data in self.results['resource_utilization'].items():
            print(f"{resource:15}: {data['used']:6.2f}/{data['available']:6.2f} "
                  f"({data['utilization_rate']*100:5.1f}%) "
                  f"Slack: {data['slack']:6.2f}")

        if self.results.get('shadow_prices'):
            print("\nShadow Prices (Value of relaxing constraints):")
            print("-" * 50)
            for constraint, price in self.results['shadow_prices'].items():
                if price and abs(price) > 1e-6:  # Only show non-zero shadow prices
                    print(f"{constraint:25}: ${price:8.2f}")

        print("\n" + "="*60)

def main():
    """Main optimization function"""
    # Initialize optimizer
    optimizer = ProductionOptimizer()

    # Load parameters
    optimizer.load_parameters('data/input_parameters.csv')

    # Formulate problem
    optimizer.formulate_problem()

    # Solve problem
    success = optimizer.solve_problem()

    if success:
        # Generate reports
        optimizer.generate_report()

        # Print summary
        optimizer.print_summary()

        print("\nOptimization completed successfully!")
        print("Check the 'outputs/' directory for detailed reports.")
    else:
        print("Optimization failed!")

if __name__ == "__main__":
    main()
