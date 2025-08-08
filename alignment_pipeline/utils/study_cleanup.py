#!/usr/bin/env python3
"""
Study Completion Verification and Cleanup

This module provides functions to verify that a study has been completed successfully
and clean up intermediate SRR files to save disk space.

A study is considered complete when:
1. The final study_counts.csv exists
2. All expected GSM samples from the mapping are present as columns
3. No SRR columns are present (indicates proper aggregation)
4. No missing or duplicate samples

After verification, intermediate SRR CSV files can be safely deleted.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def load_study_mapping(study_id: str, base_dir: str = "data") -> Optional[Dict]:
    """Load the GSM to SRR mapping for a study."""
    mapping_file = Path(base_dir) / "accessions" / f"{study_id}_mapping.json"
    
    if not mapping_file.exists():
        logger.error(f"Mapping file not found: {mapping_file}")
        return None
    
    try:
        with open(mapping_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading mapping file {mapping_file}: {e}")
        return None

def verify_study_completion(study_id: str, output_dir: str, base_dir: str = "data") -> Dict:
    """
    Verify that a study has been completed successfully.
    
    Returns:
        Dict with verification results and details
    """
    result = {
        'study_id': study_id,
        'complete': False,
        'final_file_exists': False,
        'mapping_loaded': False,
        'expected_samples': 0,
        'found_samples': 0,
        'missing_samples': [],
        'extra_samples': [],
        'srr_columns_found': [],
        'issues': []
    }
    
    # Check if final results file exists
    final_path = Path(output_dir) / f"{study_id}_counts.csv"
    if not final_path.exists():
        result['issues'].append(f"Final results file missing: {final_path}")
        return result
    
    result['final_file_exists'] = True
    
    # Load the study mapping
    study_mapping = load_study_mapping(study_id, base_dir)
    if not study_mapping:
        result['issues'].append("Could not load study mapping")
        return result
    
    result['mapping_loaded'] = True
    expected_gsms = set(study_mapping.keys())
    result['expected_samples'] = len(expected_gsms)
    
    # Load the final results file
    try:
        final_df = pd.read_csv(final_path, index_col=0)
    except Exception as e:
        result['issues'].append(f"Error reading final results: {e}")
        return result
    
    # Analyze the columns
    found_columns = set(final_df.columns)
    result['found_samples'] = len(found_columns)
    
    # Check for SRR columns (indicates incomplete aggregation)
    srr_columns = [col for col in found_columns if col.startswith('SRR') and col[3:].isdigit()]
    if srr_columns:
        result['srr_columns_found'] = srr_columns
        result['issues'].append(f"Found {len(srr_columns)} SRR columns - aggregation incomplete")
    
    # Check for missing GSM samples
    missing_gsms = expected_gsms - found_columns
    if missing_gsms:
        result['missing_samples'] = sorted(list(missing_gsms))
        result['issues'].append(f"Missing {len(missing_gsms)} expected GSM samples")
    
    # Check for extra samples (not in mapping)
    gsm_columns = [col for col in found_columns if col.startswith('GSM') and col[3:].isdigit()]
    extra_gsms = set(gsm_columns) - expected_gsms
    if extra_gsms:
        result['extra_samples'] = sorted(list(extra_gsms))
        result['issues'].append(f"Found {len(extra_gsms)} unexpected GSM samples")
    
    # Study is complete if no issues
    result['complete'] = len(result['issues']) == 0
    
    return result

def get_srr_intermediate_files(output_dir: str) -> List[Path]:
    """Get all SRR intermediate CSV files in the output directory."""
    output_path = Path(output_dir)
    return list(output_path.glob("SRR*_result.csv"))

def cleanup_srr_intermediates(output_dir: str, backup: bool = False, dry_run: bool = False) -> Dict:
    """
    Clean up SRR intermediate files after successful study completion.
    
    Args:
        output_dir: Output directory containing SRR files
        backup: Whether to create backups instead of deleting
        dry_run: If True, only show what would be done
        
    Returns:
        Dict with cleanup results
    """
    srr_files = get_srr_intermediate_files(output_dir)
    
    result = {
        'total_files': len(srr_files),
        'processed': 0,
        'errors': 0,
        'space_saved': 0,
        'dry_run': dry_run
    }
    
    if not srr_files:
        logger.info(f"No SRR intermediate files found in {output_dir}")
        return result
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Processing {len(srr_files)} SRR intermediate files")
    
    for srr_file in srr_files:
        try:
            file_size = srr_file.stat().st_size
            result['space_saved'] += file_size
            
            if not dry_run:
                if backup:
                    backup_path = srr_file.with_suffix('.csv.intermediate_backup')
                    srr_file.rename(backup_path)
                    logger.debug(f"Backed up: {srr_file.name} -> {backup_path.name}")
                else:
                    srr_file.unlink()
                    logger.debug(f"Deleted: {srr_file.name}")
            else:
                logger.debug(f"Would {'backup' if backup else 'delete'}: {srr_file.name} ({file_size:,} bytes)")
            
            result['processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing {srr_file}: {e}")
            result['errors'] += 1
    
    return result

def verify_and_cleanup_study(study_id: str, output_dir: str, base_dir: str = "data", 
                           cleanup: bool = True, backup_srr: bool = False, 
                           dry_run: bool = False) -> Dict:
    """
    Complete verification and cleanup workflow for a study.
    
    Args:
        study_id: Study identifier (e.g., 'GSE133378')
        output_dir: Directory containing study results
        base_dir: Base directory containing accessions folder
        cleanup: Whether to clean up SRR files after verification
        backup_srr: Whether to backup instead of delete SRR files
        dry_run: If True, only show what would be done
        
    Returns:
        Dict with complete results
    """
    logger.info(f"🔍 Verifying completion of study {study_id}")
    
    # Verify study completion
    verification = verify_study_completion(study_id, output_dir, base_dir)
    
    result = {
        'study_id': study_id,
        'verification': verification,
        'cleanup': None
    }
    
    # Report verification results
    if verification['complete']:
        logger.info(f"✅ Study {study_id} completed successfully:")
        logger.info(f"   📊 Expected samples: {verification['expected_samples']}")
        logger.info(f"   📊 Found samples: {verification['found_samples']}")
        logger.info(f"   🎯 All GSM samples present, no SRR columns")
        
        # Proceed with cleanup if requested
        if cleanup:
            logger.info(f"🧹 {'DRY RUN: ' if dry_run else ''}Cleaning up SRR intermediate files...")
            cleanup_result = cleanup_srr_intermediates(output_dir, backup_srr, dry_run)
            result['cleanup'] = cleanup_result
            
            if cleanup_result['total_files'] > 0:
                space_mb = cleanup_result['space_saved'] / (1024 * 1024)
                action = "Would save" if dry_run else "Saved"
                method = "backed up" if backup_srr else "deleted"
                logger.info(f"✅ {action} {space_mb:.1f} MB by {'backing up' if backup_srr else 'deleting'} {cleanup_result['processed']} files")
                if cleanup_result['errors'] > 0:
                    logger.warning(f"⚠️  {cleanup_result['errors']} files had errors during cleanup")
            else:
                logger.info("📁 No SRR intermediate files to clean up")
        else:
            logger.info("🔒 Cleanup skipped (cleanup=False)")
    else:
        logger.error(f"❌ Study {study_id} verification failed:")
        for issue in verification['issues']:
            logger.error(f"   • {issue}")
        
        if verification['missing_samples']:
            logger.error(f"   📋 Missing GSMs: {verification['missing_samples'][:5]}{'...' if len(verification['missing_samples']) > 5 else ''}")
        
        if verification['srr_columns_found']:
            logger.error(f"   📋 SRR columns found: {verification['srr_columns_found'][:5]}{'...' if len(verification['srr_columns_found']) > 5 else ''}")
        
        logger.error("🚨 Cannot proceed with cleanup - study must be completed successfully first")
    
    return result

def main():
    """Command-line interface for study verification and cleanup."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify study completion and clean up intermediate files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify a study without cleanup
  python study_cleanup.py GSE133378 --output-dir data/results/GSE133378 --no-cleanup
  
  # Verify and clean up (delete SRR files)
  python study_cleanup.py GSE133378 --output-dir data/results/GSE133378
  
  # Verify and backup SRR files instead of deleting
  python study_cleanup.py GSE133378 --output-dir data/results/GSE133378 --backup
  
  # Dry run to see what would be done
  python study_cleanup.py GSE133378 --output-dir data/results/GSE133378 --dry-run
        """
    )
    
    parser.add_argument('study_id', help='Study ID (e.g., GSE133378)')
    parser.add_argument('--output-dir', required=True, help='Directory containing study results')
    parser.add_argument('--base-dir', default='data', help='Base directory containing accessions folder')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip SRR file cleanup')
    parser.add_argument('--backup', action='store_true', help='Backup SRR files instead of deleting')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run verification and cleanup
    result = verify_and_cleanup_study(
        study_id=args.study_id,
        output_dir=args.output_dir,
        base_dir=args.base_dir,
        cleanup=not args.no_cleanup,
        backup_srr=args.backup,
        dry_run=args.dry_run
    )
    
    # Exit with appropriate code
    if result['verification']['complete']:
        print(f"\n✅ Study {args.study_id} verification and cleanup completed successfully!")
        return 0
    else:
        print(f"\n❌ Study {args.study_id} verification failed - see issues above")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
