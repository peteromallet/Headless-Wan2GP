#!/usr/bin/env python3
"""Test WanOrchestrator initialization to diagnose the error."""

import sys
import os
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger('test_init')

try:
    logger.info("=" * 60)
    logger.info("Testing WanOrchestrator initialization")
    logger.info("=" * 60)

    # Test 1: Check CUDA
    logger.info("Step 1: Checking CUDA availability...")
    import torch
    logger.info(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  Device count: {torch.cuda.device_count()}")
        logger.info(f"  Device 0: {torch.cuda.get_device_name(0)}")

    # Test 2: Try importing WanOrchestrator
    logger.info("\nStep 2: Attempting to import WanOrchestrator...")
    wan_dir = os.path.join(os.path.dirname(__file__), 'Wan2GP')
    logger.info(f"  Using Wan2GP directory: {wan_dir}")
    logger.info(f"  Directory exists: {os.path.isdir(wan_dir)}")

    # Test 3: Try the actual import
    logger.info("\nStep 3: Importing headless_wgp.WanOrchestrator...")
    from headless_wgp import WanOrchestrator

    # Test 4: Try initializing
    logger.info("\nStep 4: Initializing WanOrchestrator...")
    orchestrator = WanOrchestrator(wan_dir)

    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS: WanOrchestrator initialized successfully!")
    logger.info("=" * 60)

except Exception as e:
    logger.error("\n" + "=" * 60)
    logger.error(f"FAILED: {type(e).__name__}: {e}")
    logger.error("=" * 60)
    logger.error("\nFull traceback:", exc_info=True)
    sys.exit(1)
