"""
Memory Lifecycle Management for ADAM
Implements decay, reinforcement, and compression strategies
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import logging

# Import activity tracker
try:
    from .activity_tracker import ActivityTracker
except ImportError:
    from activity_tracker import ActivityTracker

logger = logging.getLogger(__name__)

@dataclass
class MemoryStrength:
    """Tracks memory strength and decay metrics"""
    current_strength: float = 1.0
    initial_strength: float = 1.0
    last_accessed: datetime = None
    access_count: int = 0
    reinforcement_count: int = 0
    decay_rate: float = 0.95  # Daily decay factor
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
    
    def calculate_decayed_strength(self, active_days_elapsed: int = None) -> float:
        """Calculate current strength with time decay based on active days"""
        if self.last_accessed is None:
            return self.current_strength
        
        # If active days not provided, use calendar days as fallback
        if active_days_elapsed is None:
            days_elapsed = (datetime.now() - self.last_accessed).days
        else:
            days_elapsed = active_days_elapsed
            
        if days_elapsed <= 0:
            return self.current_strength
            
        # Exponential decay: strength * (decay_rate ^ active_days)
        decayed = self.current_strength * (self.decay_rate ** days_elapsed)
        return max(0.01, decayed)  # Never fully zero
    
    def reinforce(self, boost: float = 0.1) -> float:
        """Reinforce memory when accessed"""
        self.current_strength = min(1.0, self.calculate_decayed_strength() + boost)
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.reinforcement_count += 1
        return self.current_strength


class MemoryLifecycleManager:
    """Manages memory decay, reinforcement, and compression"""
    
    # Strength thresholds
    ARCHIVE_THRESHOLD = 0.3
    COMPRESS_THRESHOLD = 0.1
    LANDMARK_THRESHOLD = 0.9
    
    # Age-based compression tiers (days)
    TIER_FULL = 7
    TIER_MODERATE = 30
    TIER_HIGH = 90
    
    def __init__(self, memory_system=None):
        self.memory_system = memory_system
        self.strength_cache: Dict[str, MemoryStrength] = {}
        self.compression_stats = {
            "total_compressed": 0,
            "storage_saved_bytes": 0,
            "compression_ratio": 0.0
        }
        
    def get_memory_strength(self, memory_id: str, metadata: Dict) -> MemoryStrength:
        """Get or create memory strength tracker"""
        if memory_id in self.strength_cache:
            return self.strength_cache[memory_id]
            
        # Initialize from metadata if available
        strength = MemoryStrength(
            current_strength=metadata.get('strength', 1.0),
            access_count=metadata.get('access_count', 0),
            last_accessed=metadata.get('last_accessed', datetime.now())
        )
        
        self.strength_cache[memory_id] = strength
        return strength
    
    def calculate_memory_importance(self, memory_id: str, metadata: Dict) -> float:
        """Calculate overall importance score for a memory"""
        strength = self.get_memory_strength(memory_id, metadata)
        
        # Factors that increase importance
        factors = {
            'strength': strength.calculate_decayed_strength(),
            'access_frequency': min(1.0, strength.access_count / 10),
            'success_rate': metadata.get('success_rate', 1.0),
            'has_code': 1.0 if metadata.get('memory_type') == 'code_pattern' else 0.5,
            'reference_count': min(1.0, metadata.get('reference_count', 0) / 5),
            'user_marked': 1.0 if metadata.get('landmark', False) else 0.0
        }
        
        # Weights for each factor
        weights = {
            'strength': 0.3,
            'access_frequency': 0.2,
            'success_rate': 0.2,
            'has_code': 0.15,
            'reference_count': 0.1,
            'user_marked': 0.05
        }
        
        # Calculate weighted importance
        importance = sum(factors[key] * weights[key] for key in factors)
        return min(1.0, importance)
    
    def classify_memory_tier(self, memory_id: str, metadata: Dict) -> str:
        """Classify memory into preservation tiers"""
        strength = self.get_memory_strength(memory_id, metadata)
        current_strength = strength.calculate_decayed_strength()
        importance = self.calculate_memory_importance(memory_id, metadata)
        
        # Never compress landmarks
        if metadata.get('landmark', False) or importance >= self.LANDMARK_THRESHOLD:
            return 'landmark'
        
        # Check strength thresholds
        if current_strength < self.COMPRESS_THRESHOLD:
            return 'compress_ultra'
        elif current_strength < self.ARCHIVE_THRESHOLD:
            return 'compress_high'
        
        # Check age-based tiers
        age_days = (datetime.now() - strength.last_accessed).days
        if age_days > self.TIER_HIGH:
            return 'compress_high'
        elif age_days > self.TIER_MODERATE:
            return 'compress_moderate'
        elif age_days > self.TIER_FULL:
            return 'archive'
        
        return 'active'
    
    async def apply_decay_to_all_memories(self):
        """Apply decay to all memories in the system"""
        if not self.memory_system:
            return
            
        logger.info("Applying decay to all memories...")
        
        # Get all memories
        all_memories = self.memory_system.collection.get()
        
        updated_count = 0
        archived_count = 0
        compress_candidates = []
        
        for i, (memory_id, metadata) in enumerate(zip(all_memories['ids'], all_memories['metadatas'])):
            strength = self.get_memory_strength(memory_id, metadata)
            old_strength = strength.current_strength
            new_strength = strength.calculate_decayed_strength()
            
            # Update metadata
            metadata['strength'] = new_strength
            metadata['last_decay_applied'] = datetime.now().isoformat()
            
            # Classify tier
            tier = self.classify_memory_tier(memory_id, metadata)
            metadata['memory_tier'] = tier
            
            # Track changes
            if tier == 'archive':
                archived_count += 1
            elif tier.startswith('compress'):
                compress_candidates.append((memory_id, tier))
            
            # Update in database
            self.memory_system.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )
            
            updated_count += 1
            
        logger.info(f"Decay applied to {updated_count} memories")
        logger.info(f"Archived {archived_count} memories")
        logger.info(f"Identified {len(compress_candidates)} candidates for compression")
        
        return compress_candidates
    
    def reinforce_memory(self, memory_id: str, metadata: Dict, boost: float = 0.1):
        """Reinforce a memory when accessed"""
        strength = self.get_memory_strength(memory_id, metadata)
        new_strength = strength.reinforce(boost)
        
        # Update metadata
        metadata['strength'] = new_strength
        metadata['access_count'] = strength.access_count
        metadata['last_accessed'] = strength.last_accessed.isoformat()
        
        # Update tier
        metadata['memory_tier'] = self.classify_memory_tier(memory_id, metadata)
        
        logger.debug(f"Reinforced memory {memory_id}: strength {new_strength:.2f}")
        return new_strength
    
    async def compress_memory(self, memory_id: str, content: str, metadata: Dict, compression_level: str) -> Tuple[str, Dict]:
        """Compress a memory based on its tier"""
        original_length = len(content)
        
        if compression_level == 'compress_moderate':
            # Keep key exchanges, remove fluff
            compressed = await self._moderate_compression(content, metadata)
        elif compression_level == 'compress_high':
            # Extract only insights
            compressed = await self._high_compression(content, metadata)
        elif compression_level == 'compress_ultra':
            # Single paragraph summary
            compressed = await self._ultra_compression(content, metadata)
        else:
            return content, metadata
        
        # Update metadata
        compressed_length = len(compressed)
        compression_ratio = 1 - (compressed_length / original_length)
        
        metadata['compressed'] = True
        metadata['compression_level'] = compression_level
        metadata['compression_ratio'] = compression_ratio
        metadata['original_length'] = original_length
        metadata['compressed_at'] = datetime.now().isoformat()
        
        # Update stats
        self.compression_stats['total_compressed'] += 1
        self.compression_stats['storage_saved_bytes'] += (original_length - compressed_length)
        
        logger.info(f"Compressed memory {memory_id}: {compression_ratio:.1%} reduction")
        
        return compressed, metadata
    
    async def _moderate_compression(self, content: str, metadata: Dict) -> str:
        """Remove redundancy, keep substance"""
        # For now, simple implementation - can be enhanced with LLM
        lines = content.split('\n')
        
        # Keep lines with key indicators
        key_indicators = ['error', 'solution', 'fixed', 'query', 'result', 'code', 'def', 'class']
        important_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in key_indicators):
                important_lines.append(line)
        
        # If too aggressive, keep at least 50% of content
        if len(important_lines) < len(lines) * 0.5:
            important_lines = lines[::2]  # Keep every other line
        
        return '\n'.join(important_lines)
    
    async def _high_compression(self, content: str, metadata: Dict) -> str:
        """Extract key insights only"""
        # Extract problem/solution pairs
        insights = []
        
        # Simple pattern matching - enhance with LLM later
        if metadata.get('query_text'):
            insights.append(f"Q: {metadata['query_text']}")
        
        # Look for solution patterns
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ['solution:', 'answer:', 'fixed:', 'result:']):
                insights.append(line)
                # Include next 2 lines for context
                if i + 1 < len(lines):
                    insights.append(lines[i + 1])
                if i + 2 < len(lines):
                    insights.append(lines[i + 2])
        
        return '\n'.join(insights) if insights else f"Memory from {metadata.get('timestamp', 'unknown time')}"
    
    async def _ultra_compression(self, content: str, metadata: Dict) -> str:
        """Create single paragraph summary"""
        # Ultra simple for now - just metadata summary
        summary_parts = []
        
        if metadata.get('query_text'):
            summary_parts.append(f"Query: {metadata['query_text'][:50]}...")
        
        if metadata.get('memory_type'):
            summary_parts.append(f"Type: {metadata['memory_type']}")
        
        if metadata.get('topics'):
            summary_parts.append(f"Topics: {', '.join(metadata['topics'][:3])}")
        
        summary_parts.append(f"Original length: {metadata.get('original_length', len(content))} chars")
        
        return ' | '.join(summary_parts)
    
    def get_lifecycle_stats(self) -> Dict:
        """Get statistics about memory lifecycle"""
        if not self.memory_system:
            return {}
        
        all_memories = self.memory_system.collection.get()
        
        tiers = {
            'active': 0,
            'archive': 0,
            'landmark': 0,
            'compressed': 0
        }
        
        strength_distribution = []
        
        for metadata in all_memories['metadatas']:
            tier = metadata.get('memory_tier', 'unknown')
            if tier in tiers:
                tiers[tier] += 1
            elif tier.startswith('compress'):
                tiers['compressed'] += 1
            
            if 'strength' in metadata:
                strength_distribution.append(metadata['strength'])
        
        return {
            'tier_distribution': tiers,
            'average_strength': np.mean(strength_distribution) if strength_distribution else 0,
            'compression_stats': self.compression_stats,
            'total_memories': len(all_memories['ids'])
        }


# Scheduled task for automatic decay
async def run_decay_cycle(memory_system):
    """Run a decay cycle on all memories"""
    manager = MemoryLifecycleManager(memory_system)
    
    while True:
        try:
            # Run decay once per day
            await asyncio.sleep(86400)  # 24 hours
            
            logger.info("Running scheduled memory decay cycle...")
            compress_candidates = await manager.apply_decay_to_all_memories()
            
            # Compress memories if needed (simplified for now)
            if compress_candidates:
                logger.info(f"Found {len(compress_candidates)} memories to compress")
                # TODO: Implement batch compression with LLM
            
        except Exception as e:
            logger.error(f"Error in decay cycle: {e}")
            await asyncio.sleep(3600)  # Retry in 1 hour