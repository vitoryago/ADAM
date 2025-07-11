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

# Import activity tracker and compressor
try:
    from .activity_tracker import ActivityTracker
    from .memory_compressor import MemoryCompressor, CompressionResult
except ImportError:
    from activity_tracker import ActivityTracker
    from memory_compressor import MemoryCompressor, CompressionResult

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
        
        # Initialize activity tracker
        persist_dir = getattr(memory_system, 'persist_directory', './adam_memory_advanced')
        self.activity_tracker = ActivityTracker(persist_dir)
        
        # Initialize memory compressor
        self.compressor = MemoryCompressor()
        
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
        
        # Calculate active days elapsed for this memory
        timestamp = metadata.get('timestamp')
        if timestamp:
            memory_date = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
            active_days = self.activity_tracker.calculate_active_age_days(memory_date)
        else:
            active_days = 0
        
        # Factors that increase importance
        factors = {
            'strength': strength.calculate_decayed_strength(active_days),
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
        
        # Calculate active days for this memory
        timestamp = metadata.get('timestamp')
        if timestamp:
            memory_date = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
            active_days = self.activity_tracker.calculate_active_age_days(memory_date)
        else:
            active_days = 0
            
        current_strength = strength.calculate_decayed_strength(active_days)
        importance = self.calculate_memory_importance(memory_id, metadata)
        
        # Never compress landmarks
        if metadata.get('landmark', False) or importance >= self.LANDMARK_THRESHOLD:
            return 'landmark'
        
        # Check strength thresholds
        if current_strength < self.COMPRESS_THRESHOLD:
            return 'compress_ultra'
        elif current_strength < self.ARCHIVE_THRESHOLD:
            return 'compress_high'
        
        # Check age-based tiers using active days
        if active_days > self.TIER_HIGH:
            return 'compress_high'
        elif active_days > self.TIER_MODERATE:
            return 'compress_moderate'
        elif active_days > self.TIER_FULL:
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
        try:
            # Map tier names to compression levels
            level_mapping = {
                'compress_moderate': 'moderate',
                'compress_high': 'high',
                'compress_ultra': 'ultra'
            }
            
            level = level_mapping.get(compression_level, 'moderate')
            
            # Use LLM-based compression
            result = await self.compressor.compress_memory(content, metadata, level)
            
            # Update metadata with compression info
            metadata['compressed'] = True
            metadata['compression_level'] = compression_level
            metadata['compression_ratio'] = result.compression_ratio
            metadata['original_length'] = len(content)
            metadata['compressed_length'] = len(result.compressed_content)
            metadata['compressed_at'] = datetime.now().isoformat()
            metadata['preserved_elements'] = result.preserved_elements
            metadata['removed_elements'] = result.removed_elements
            metadata['tokens_saved'] = result.tokens_saved
            
            # Update stats
            self.compression_stats['total_compressed'] += 1
            self.compression_stats['storage_saved_bytes'] += (len(content) - len(result.compressed_content))
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['storage_saved_bytes'] / 
                (self.compression_stats['storage_saved_bytes'] + len(result.compressed_content))
                if self.compression_stats['total_compressed'] > 0 else 0
            )
            
            logger.info(f"Compressed memory {memory_id}: {result.compression_ratio:.1%} reduction")
            logger.info(f"  Preserved: {', '.join(result.preserved_elements)}")
            logger.info(f"  Tokens saved: {result.tokens_saved}")
            
            return result.compressed_content, metadata
            
        except Exception as e:
            logger.error(f"Failed to compress memory {memory_id}: {e}")
            # Return original on failure
            return content, metadata
    
    
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