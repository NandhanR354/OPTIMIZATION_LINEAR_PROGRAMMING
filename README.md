# Task 4: Optimization Model - Linear Programming with PuLP

## Overview
This project demonstrates solving a business optimization problem using linear programming techniques with Python's PuLP library. The project focuses on a production planning optimization problem.

## Problem Statement
A manufacturing company produces multiple products with limited resources. The goal is to maximize profit while satisfying production constraints including:
- Raw material availability
- Production capacity limits
- Minimum demand requirements
- Storage limitations

## Features
- Linear programming model formulation
- Multiple constraint types (resource, demand, capacity)
- Sensitivity analysis
- Scenario planning
- Comprehensive reporting and insights
- Interactive Jupyter notebook analysis

## Project Structure
```
Task4_Optimization/
├── data/
│   ├── raw/
│   └── input_parameters.csv
├── src/
│   ├── __init__.py
│   ├── optimization_model.py
│   ├── data_loader.py
│   ├── report_generator.py
│   └── sensitivity_analysis.py
├── outputs/
│   ├── optimization_results.json
│   ├── sensitivity_report.html
│   └── production_plan.csv
├── notebooks/
│   └── optimization_analysis.ipynb
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic Optimization
```bash
python src/optimization_model.py
```

### 2. Sensitivity Analysis
```bash
python src/sensitivity_analysis.py
```

### 3. Interactive Analysis
```bash
jupyter notebook notebooks/optimization_analysis.ipynb
```

## Business Problem
**Manufacturing Production Planning**

A company manufactures three products (A, B, C) with the following characteristics:
- Product A: Profit $40/unit, requires 2 hours labor, 3 kg raw material
- Product B: Profit $30/unit, requires 1 hour labor, 2 kg raw material  
- Product C: Profit $50/unit, requires 3 hours labor, 1 kg raw material

Constraints:
- Available labor: 100 hours/week
- Available raw material: 80 kg/week
- Minimum demand: Product A ≥ 5, Product B ≥ 10, Product C ≥ 8
- Storage capacity: Maximum 50 units total

## Key Insights
- Optimal production mix maximizing profit
- Shadow prices for constraint relaxation
- Bottleneck resource identification
- Sensitivity ranges for parameters

## Author
CODTECH Internship - Data Science Track
