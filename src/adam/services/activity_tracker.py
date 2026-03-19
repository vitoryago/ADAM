"""
Activity Tracker for ADAM
Tracks active usage days to properly calculate memory age
"""
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ActivityTracker:
    """
    Tracks ADAM's active days to calculate proper memory age.
    Only days with actual usage count towards memory decay.
    """

    def __init__(self, persist_directory: str = "./adam_memory_advanced"):
        self.persist_directory = Path(persist_directory)
        self.activity_file = self.persist_directory / "activity_log.json"
        self.activity_data = self._load_activity_data()

    def _load_activity_data(self) -> Dict:
        """Load activity data from disk"""
        if self.activity_file.exists():
            try:
                with open(self.activity_file, 'r') as f:
                    data = json.load(f)
                    return {
                        "daily_activity": data.get("daily_activity", {}),
                        "active_days": data.get("active_days", []),
                        "first_activity": data.get("first_activity"),
                        "last_activity": data.get("last_activity"),
                        "total_interactions": data.get("total_interactions", 0)
                    }
            except Exception as e:
                logger.error(f"Error loading activity data: {e}")
                return self._default_activity_data()
        return self._default_activity_data()

    def _default_activity_data(self) -> Dict:
        """Default activity data structure"""
        return {
            "daily_activity": {},
            "active_days": [],
            "first_activity": None,
            "last_activity": None,
            "total_interactions": 0
        }

    def _save_activity_data(self):
        """Save activity data to disk"""
        try:
            with open(self.activity_file, 'w') as f:
                json.dump(self.activity_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving activity data: {e}")

    def record_interaction(self):
        """Record a new interaction for today"""
        today = date.today().isoformat()

        if today not in self.activity_data["daily_activity"]:
            self.activity_data["daily_activity"][today] = 0
            self.activity_data["active_days"].append(today)
            logger.info(f"New active day recorded: {today}")

        self.activity_data["daily_activity"][today] += 1
        self.activity_data["total_interactions"] += 1

        if not self.activity_data["first_activity"]:
            self.activity_data["first_activity"] = today
        self.activity_data["last_activity"] = today

        self._save_activity_data()

        return self.get_active_day_index(today)

    def get_active_day_index(self, date_str: Optional[str] = None) -> int:
        """
        Get the active day index for a given date.
        This is the number of active days since first interaction.
        """
        if not self.activity_data["active_days"]:
            return 0

        if date_str is None:
            date_str = date.today().isoformat()

        try:
            return self.activity_data["active_days"].index(date_str)
        except ValueError:
            return len(self.activity_data["active_days"])

    def calculate_active_age_days(self, from_date: datetime) -> int:
        """
        Calculate how many active days have passed since a given date.
        Only counts days where ADAM was actually used.
        """
        from_date_str = from_date.date().isoformat()

        if not self.activity_data["active_days"]:
            return 0

        active_days_count = 0
        for active_day in self.activity_data["active_days"]:
            if active_day > from_date_str:
                active_days_count += 1

        return active_days_count

    def get_days_since_last_activity(self) -> int:
        """Get calendar days since last activity"""
        if not self.activity_data["last_activity"]:
            return 0

        last_date = datetime.fromisoformat(self.activity_data["last_activity"]).date()
        return (date.today() - last_date).days

    def get_activity_summary(self) -> Dict:
        """Get a summary of activity patterns"""
        if not self.activity_data["active_days"]:
            return {
                "status": "No activity recorded",
                "total_active_days": 0,
                "total_interactions": 0
            }

        total_active_days = len(self.activity_data["active_days"])
        avg_interactions = self.activity_data["total_interactions"] / total_active_days

        most_active_day = max(
            self.activity_data["daily_activity"].items(),
            key=lambda x: x[1]
        ) if self.activity_data["daily_activity"] else (None, 0)

        return {
            "first_activity": self.activity_data["first_activity"],
            "last_activity": self.activity_data["last_activity"],
            "total_active_days": total_active_days,
            "total_interactions": self.activity_data["total_interactions"],
            "avg_interactions_per_day": round(avg_interactions, 1),
            "most_active_day": most_active_day[0],
            "most_active_day_count": most_active_day[1],
            "days_since_last_activity": self.get_days_since_last_activity(),
            "current_active_day_index": self.get_active_day_index()
        }

    def get_activity_pattern(self, last_n_days: int = 30) -> Dict[str, int]:
        """Get activity pattern for the last N active days"""
        if not self.activity_data["active_days"]:
            return {}

        recent_days = self.activity_data["active_days"][-last_n_days:]

        return {
            day: self.activity_data["daily_activity"].get(day, 0)
            for day in recent_days
        }
