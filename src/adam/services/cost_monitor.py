#!/usr/bin/env python3
"""
Real-time Cost Monitoring Dashboard for ADAM
Tracks costs, provides alerts, and shows usage analytics
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class CostMonitor:
    """
    Monitors and tracks LLM usage costs in real-time
    """

    def __init__(self, storage_path: str = "./cost_tracking"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Current session tracking
        self.current_session = {
            "start_time": datetime.now(),
            "queries": [],
            "total_cost": 0.0,
            "model_usage": defaultdict(int),
            "model_costs": defaultdict(float)
        }

        # Daily tracking
        self.daily_file = self.storage_path / f"daily_{datetime.now().strftime('%Y%m%d')}.json"
        self.load_daily_data()

        # Alerts
        self.alerts = {
            "high_cost_query": 0.10,
            "daily_warning": 0.80,
            "daily_limit": 1.00,
            "hourly_spike": 0.25
        }

    def track_query(self,
                   query: str,
                   model: str,
                   input_tokens: int,
                   output_tokens: int,
                   actual_cost: float,
                   response_time: float):
        """
        Track a single query's cost and performance
        """
        query_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": actual_cost,
            "response_time": response_time
        }

        # Update session
        self.current_session["queries"].append(query_data)
        self.current_session["total_cost"] += actual_cost
        self.current_session["model_usage"][model] += 1
        self.current_session["model_costs"][model] += actual_cost

        # Update daily data
        self.daily_data["queries"].append(query_data)
        self.daily_data["total_cost"] += actual_cost
        self.daily_data["model_costs"][model] += actual_cost

        # Check alerts
        self._check_alerts(query_data)

        # Save periodically
        if len(self.current_session["queries"]) % 10 == 0:
            self.save_daily_data()

        logger.info(f"Tracked: {model} query, ${actual_cost:.4f}, "
                   f"Daily total: ${self.daily_data['total_cost']:.4f}")

    def _check_alerts(self, query_data: Dict):
        """Check and trigger cost alerts"""
        if query_data["cost"] > self.alerts["high_cost_query"]:
            logger.warning(f"High cost query: ${query_data['cost']:.4f} for {query_data['model']}")

        daily_total = self.daily_data["total_cost"]
        daily_limit = self.alerts["daily_limit"]

        if daily_total >= daily_limit:
            logger.error(f"DAILY LIMIT REACHED: ${daily_total:.4f} >= ${daily_limit}")
        elif daily_total >= self.alerts["daily_warning"]:
            remaining = daily_limit - daily_total
            logger.warning(f"Approaching daily limit: ${remaining:.4f} remaining")

        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hour_queries = [q for q in self.daily_data["queries"]
                       if datetime.fromisoformat(q["timestamp"]).replace(minute=0, second=0, microsecond=0) == current_hour]
        hour_cost = sum(q["cost"] for q in hour_queries)

        if hour_cost > self.alerts["hourly_spike"]:
            logger.warning(f"High hourly spend: ${hour_cost:.4f} in current hour")

    def load_daily_data(self):
        """Load or create daily tracking data"""
        if self.daily_file.exists():
            with open(self.daily_file, 'r') as f:
                self.daily_data = json.load(f)
        else:
            self.daily_data = {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "queries": [],
                "total_cost": 0.0,
                "model_costs": defaultdict(float)
            }

    def save_daily_data(self):
        """Save daily tracking data"""
        save_data = {
            "date": self.daily_data["date"],
            "queries": self.daily_data["queries"],
            "total_cost": self.daily_data["total_cost"],
            "model_costs": dict(self.daily_data["model_costs"])
        }

        with open(self.daily_file, 'w') as f:
            json.dump(save_data, f, indent=2)

    def get_current_status(self) -> Dict:
        """Get current cost status"""
        daily_limit = self.alerts["daily_limit"]
        daily_total = self.daily_data["total_cost"]

        return {
            "daily_total": daily_total,
            "daily_limit": daily_limit,
            "daily_remaining": max(0, daily_limit - daily_total),
            "daily_percentage": (daily_total / daily_limit) * 100,
            "queries_today": len(self.daily_data["queries"]),
            "average_cost": daily_total / max(1, len(self.daily_data["queries"])),
            "model_breakdown": dict(self.daily_data["model_costs"]),
            "is_limited": daily_total >= daily_limit
        }

    def get_hourly_breakdown(self) -> Dict[str, float]:
        """Get hourly cost breakdown for today"""
        hourly_costs = defaultdict(float)

        for query in self.daily_data["queries"]:
            hour = datetime.fromisoformat(query["timestamp"]).strftime('%H:00')
            hourly_costs[hour] += query["cost"]

        return dict(hourly_costs)

    def generate_cost_report(self, days: int = 7) -> Dict:
        """Generate cost report for past N days"""
        report = {
            "period": f"Last {days} days",
            "daily_costs": {},
            "total_cost": 0.0,
            "model_breakdown": defaultdict(float),
            "query_count": 0,
            "expensive_queries": []
        }

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            file_path = self.storage_path / f"daily_{date}.json"

            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)

                report["daily_costs"][data["date"]] = data["total_cost"]
                report["total_cost"] += data["total_cost"]
                report["query_count"] += len(data["queries"])

                for model, cost in data.get("model_costs", {}).items():
                    report["model_breakdown"][model] += cost

                for query in data["queries"]:
                    if query["cost"] > 0.05:
                        report["expensive_queries"].append({
                            "date": data["date"],
                            "query": query["query"],
                            "model": query["model"],
                            "cost": query["cost"]
                        })

        report["expensive_queries"].sort(key=lambda x: x["cost"], reverse=True)
        report["expensive_queries"] = report["expensive_queries"][:10]

        return report

    def visualize_costs(self, save_path: Optional[str] = None):
        """Create cost visualization dashboard"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for cost visualization")
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Hourly costs today
        hourly = self.get_hourly_breakdown()
        hours = sorted(hourly.keys())
        costs = [hourly[h] for h in hours]

        ax1.bar(hours, costs, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Cost ($)')
        ax1.set_title('Hourly Costs Today')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=self.alerts["hourly_spike"], color='red', linestyle='--', label='Alert threshold')
        ax1.legend()

        # 2. Model usage breakdown (pie chart)
        model_costs = dict(self.daily_data["model_costs"])
        if model_costs:
            models = list(model_costs.keys())
            costs = list(model_costs.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

            ax2.pie(costs, labels=models, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Cost by Model (Today)')

        # 3. Daily trend (last 7 days)
        report = self.generate_cost_report(7)
        dates = sorted(report["daily_costs"].keys())
        daily_costs = [report["daily_costs"][d] for d in dates]

        ax3.plot(dates, daily_costs, marker='o', linewidth=2, markersize=8)
        ax3.fill_between(range(len(dates)), daily_costs, alpha=0.3)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Daily Cost ($)')
        ax3.set_title('7-Day Cost Trend')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=self.alerts["daily_limit"], color='red', linestyle='--', label='Daily limit')
        ax3.legend()

        # 4. Current status gauge
        status = self.get_current_status()
        ax4.clear()

        wedges, texts = ax4.pie([status["daily_percentage"], 100 - status["daily_percentage"]],
                               startangle=90,
                               colors=['red' if status["daily_percentage"] > 80 else 'green', 'lightgray'])

        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax4.add_artist(centre_circle)

        ax4.text(0, 0, f'${status["daily_total"]:.2f}\nof\n${status["daily_limit"]:.2f}',
                ha='center', va='center', fontsize=14, weight='bold')
        ax4.set_title('Daily Budget Usage')

        plt.suptitle(f'ADAM Cost Monitoring Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    fontsize=16, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved cost dashboard to {save_path}")
        else:
            plt.show()

        return fig


# Global cost monitor instance
_cost_monitor = None

def get_cost_monitor() -> CostMonitor:
    """Get or create the global cost monitor"""
    global _cost_monitor
    if _cost_monitor is None:
        _cost_monitor = CostMonitor()
    return _cost_monitor
