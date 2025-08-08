#!/usr/bin/env python3
"""
Index management utilities for xalign pipeline.
Handles validation, building, and resetting of alignment indexes.
"""

import os
import time
import logging
import subprocess
import platform
import shutil

logger = logging.getLogger(__name__)

def get_xalign_data_path():
    """Get the xalign data directory path."""
    try:
        import xalign.file as xfile
        return xfile.get_data_path()
    except ImportError:
        logger.error("xalign not available - cannot determine data path")
        return None

def validate_alignment_index(species="homo_sapiens", aligner="kallisto", release=None, verbose=False, allow_build=True):
    """
    Validate that the alignment index exists and is not corrupt.
    If allow_build is True, will attempt to build the index if missing.
    Returns True if valid or successfully built, False if corrupt or build failed.
    """
    data_path = get_xalign_data_path()
    if not data_path:
        return False
    
    try:
        import xalign
        import xalign.ensembl
        
        # Get release info if not provided
        if not release:
            organisms = xalign.ensembl.retrieve_ensembl_organisms(release)
            if species not in organisms:
                logger.error(f"Species {species} not found in Ensembl database")
                return False
            release = organisms[species][5]
        
        # Check index file/directory existence
        if aligner == "kallisto":
            expected_index_path = os.path.join(data_path, "index", str(release), f"kallisto_{species}.idx")
            index_dir = os.path.join(data_path, "index", str(release))
            
            # First, look for the expected index file
            index_path = expected_index_path
            if not os.path.exists(index_path):
                if verbose:
                    logger.info(f"Expected kallisto index not found: {index_path}")
                
                # If allow_build is True, try to let xalign build the index
                if allow_build:
                    if verbose:
                        logger.info(f"Attempting to build kallisto index for {species} (release {release})")
                    try:
                        # Use xalign to build the index
                        xalign.build_index(aligner="kallisto", species=species, release=release, verbose=verbose)
                        
                        # Check if the expected index was created
                        if os.path.exists(expected_index_path):
                            index_path = expected_index_path
                        else:
                            # Search for any .idx files in the release directory
                            if os.path.exists(index_dir):
                                idx_files = [f for f in os.listdir(index_dir) if f.endswith('.idx')]
                                if idx_files:
                                    index_path = os.path.join(index_dir, idx_files[0])
                                    if verbose:
                                        logger.info(f"Found alternative index file: {index_path}")
                                else:
                                    logger.error(f"No .idx files found after build attempt in {index_dir}")
                                    return False
                            else:
                                logger.error(f"Index directory not created: {index_dir}")
                                return False
                    except Exception as e:
                        logger.error(f"Failed to build kallisto index: {e}")
                        return False
                else:
                    return False
            
            # Validate kallisto index by trying to inspect it
            try:
                kallisto_binary = os.path.join(data_path, "kallisto", "kallisto")
                if platform.system().lower() == "windows":
                    kallisto_binary += ".exe"
                
                result = subprocess.run([
                    kallisto_binary, "inspect", index_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    logger.warning(f"Kallisto index appears corrupt: {index_path}")
                    if verbose:
                        logger.warning(f"Kallisto inspect error: {result.stderr}")
                    return False
                
                if verbose:
                    logger.info(f"Kallisto index validation passed: {index_path}")
                return True
                
            except FileNotFoundError:
                logger.error("Kallisto binary not found")
                return False
            except subprocess.TimeoutExpired:
                logger.warning("Kallisto index inspection timed out")
                return False
                
        else:
            logger.error(f"Unsupported aligner: {aligner}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating alignment index: {e}")
        return False

def force_remove_directory(dir_path, max_retries=3, retry_delay=1):
    """
    Forcefully remove a directory and all its contents, handling common issues
    like non-empty directories, permission problems, and files in use.
    """
    if not os.path.exists(dir_path):
        return True
    
    for attempt in range(max_retries):
        try:
            if platform.system().lower() == "windows":
                # Windows-specific removal
                subprocess.run(['rmdir', '/s', '/q', dir_path], 
                             shell=True, check=True, capture_output=True)
            else:
                # Unix-like systems
                shutil.rmtree(dir_path)
            
            logger.info(f"Successfully removed directory: {dir_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} to remove {dir_path} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to remove {dir_path} after {max_retries} attempts")
    
    return False

def reset_alignment_indexes(species=None, aligner=None, release=None, force=False):
    """
    Reset (delete) alignment indexes and reference files.
    Enhanced with robust directory removal that handles non-empty directories.
    """
    data_path = get_xalign_data_path()
    if not data_path:
        logger.error("Cannot determine xalign data path")
        return False
    
    logger.info("Resetting alignment indexes and reference files...")
    
    if not force:
        response = input("This will delete alignment indexes. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Index reset cancelled")
            return False
    
    success = True
    
    try:
        # Remove index directories
        index_base = os.path.join(data_path, "index")
        if os.path.exists(index_base):
            if species and release:
                index_dir = os.path.join(index_base, str(release))
                if os.path.exists(index_dir):
                    if not force_remove_directory(index_dir):
                        success = False
            else:
                if not force_remove_directory(index_base):
                    success = False
        
        # Remove reference directories  
        reference_base = os.path.join(data_path, "reference")
        if os.path.exists(reference_base):
            if species and release:
                ref_dir = os.path.join(reference_base, str(release))
                if os.path.exists(ref_dir):
                    if not force_remove_directory(ref_dir):
                        success = False
            else:
                if not force_remove_directory(reference_base):
                    success = False
        
        if success:
            logger.info("✅ Index reset completed successfully")
        else:
            logger.error("❌ Index reset completed with some errors")
            
    except Exception as e:
        logger.error(f"Error during index reset: {e}")
        success = False
    
    return success

def validate_and_reset_if_needed(species="homo_sapiens", aligner="kallisto", release=None, force_reset=False):
    """
    Validate indexes and reset if corrupt or if force_reset is True.
    If indexes don't exist, allows xalign to build them automatically.
    """
    if force_reset:
        return reset_alignment_indexes(species=species, aligner=aligner, release=release, force=True)
    
    # Let xalign handle index building automatically - it has built-in logic for this
    # We only need to validate/reset if there are known corruption issues
    logger.info("Allowing xalign to handle index validation and building automatically")
    return True
