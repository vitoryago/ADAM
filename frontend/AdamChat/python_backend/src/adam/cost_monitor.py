"""
Cost monitoring and budget management for ADAM
"""
import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelPricing:
    """Pricing information for different models"""
    input_cost_per_1k: float
    output_cost_per_1k: float
    name: str

class CostMonitor:
    """Monitor and track costs across all ADAM operations"""
    
    def __init__(self, config):
        self.config = config
        self.pricing = {
            'grok-4': ModelPricing(0.01, 0.02, 'Grok-4'),
            'grok-3-mini-fast': ModelPricing(0.001, 0.002, 'Grok-3-Mini-Fast'),
            'gpt-4o-mini': ModelPricing(0.000150, 0.000600, 'GPT-4o-Mini'),
            'gpt-4o': ModelPricing(0.005, 0.015, 'GPT-4o'),
            'claude-3-haiku': ModelPricing(0.00025, 0.00125, 'Claude-3-Haiku'),
            'claude-3-sonnet': ModelPricing(0.003, 0.015, 'Claude-3-Sonnet')
        }
        
        # Storage for cost tracking
        self.cost_file = Path(config.conversation_storage_path) / 'cost_tracking.json'
        self.cost_data = self._load_cost_data()
    
    def _load_cost_data(self) -> Dict[str, Any]:
        """Load cost tracking data from file"""
        if self.cost_file.exists():
            try:
                with open(self.cost_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            'daily_costs': {},
            'monthly_costs': {},
            'model_usage': {},
            'query_count': 0,
            'last_reset': datetime.now().isoformat()
        }
    
    def _save_cost_data(self):
        """Save cost tracking data to file"""
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cost_file, 'w') as f:
            json.dump(self.cost_data, f, indent=2)
    
    async def track_query(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Track the cost of a query"""
        today = datetime.now().strftime('%Y-%m-%d')
        this_month = datetime.now().strftime('%Y-%m')
        
        # Calculate cost
        cost = 0.0
        if model in self.pricing:
            pricing = self.pricing[model]
            cost = (input_tokens / 1000 * pricing.input_cost_per_1k + 
                   output_tokens / 1000 * pricing.output_cost_per_1k)
        
        # Update tracking data
        self.cost_data['daily_costs'][today] = self.cost_data['daily_costs'].get(today, 0.0) + cost
        self.cost_data['monthly_costs'][this_month] = self.cost_data['monthly_costs'].get(this_month, 0.0) + cost
        self.cost_data['model_usage'][model] = self.cost_data['model_usage'].get(model, 0.0) + cost
        self.cost_data['query_count'] += 1
        
        # Save data
        self._save_cost_data()
        
        # Check budget limits
        daily_cost = self.cost_data['daily_costs'].get(today, 0.0)
        monthly_cost = self.cost_data['monthly_costs'].get(this_month, 0.0)
        
        warnings = []
        if daily_cost > self.config.daily_cost_limit:
            warnings.append(f"Daily cost limit exceeded: ${daily_cost:.4f}")
        if monthly_cost > self.config.monthly_cost_limit:
            warnings.append(f"Monthly cost limit exceeded: ${monthly_cost:.4f}")
        
        return {
            'cost': cost,
            'daily_cost': daily_cost,
            'monthly_cost': monthly_cost,
            'warnings': warnings,
            'model': model,
            'tokens_used': input_tokens + output_tokens
        }
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        today = datetime.now().strftime('%Y-%m-%d')
        this_month = datetime.now().strftime('%Y-%m')
        
        return {
            'dailyCost': self.cost_data['daily_costs'].get(today, 0.0),
            'monthlyCost': self.cost_data['monthly_costs'].get(this_month, 0.0),
            'queryCount': self.cost_data['query_count'],
            'modelUsage': self.cost_data['model_usage'].copy(),
            'dailyLimit': self.config.daily_cost_limit,
            'monthlyLimit': self.config.monthly_cost_limit
        }
    
    def get_optimal_model(self, complexity: str = 'simple') -> str:
        """Get the most cost-effective model for the given complexity"""
        if complexity == 'complex':
            return self.config.default_complex_model
        elif complexity == 'coding':
            return self.config.default_coding_model
        else:
            return self.config.default_simple_model