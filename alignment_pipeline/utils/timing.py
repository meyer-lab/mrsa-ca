#!/usr/bin/env python3
"""
Simple timing utilities for the RNA-seq processing pipeline.
"""

import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class SimpleTimer:
    """Simplified timing utility for tracking pipeline performance."""
    
    def __init__(self):
        self.timers = {}
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.pipeline_start = time.time()
    
    def start(self, name, item_id=None):
        """Start a timer."""
        key = f"{name}_{item_id}" if item_id else name
        self.timers[key] = time.time()
    
    def end(self, name, item_id=None):
        """End a timer and log the result."""
        key = f"{name}_{item_id}" if item_id else name
        
        if key not in self.timers:
            logger.warning(f"Timer {key} was not started")
            return 0
        
        elapsed = time.time() - self.timers[key]
        del self.timers[key]
        
        # Update statistics
        self.totals[name] += elapsed
        self.counts[name] += 1
        
        # Log timing
        avg = self.totals[name] / self.counts[name]
        logger.info(f"⏱️  {name.upper()}: {elapsed:.1f}s (avg: {avg:.1f}s, count: {self.counts[name]})")
        
        return elapsed
    
    def summary(self):
        """Log a timing summary."""
        total_time = time.time() - self.pipeline_start
        
        logger.info("=" * 50)
        logger.info(f"🏁 TIMING SUMMARY - Total: {total_time/60:.1f}m")
        
        for name in ['download', 'alignment']:
            if name in self.totals:
                total = self.totals[name]
                count = self.counts[name]
                avg = total / count if count > 0 else 0
                logger.info(f"  {name.upper()}: {count} items, {total/60:.1f}m total, {avg:.1f}s avg")
        
        logger.info("=" * 50)

# Global timer instance
timer = SimpleTimer()
