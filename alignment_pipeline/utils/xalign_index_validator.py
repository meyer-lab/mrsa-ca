#!/usr/bin/env python3
"""
XAlign index validation and monitoring utility.

This module provides functions to check XAlign index status, integrity,
and storage locations. It helps prevent reference genome index corruption
issues by monitoring index health and providing detailed diagnostic information.
"""

import logging
import os
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_xalign_data_path() -> str:
    """
    Get the XAlign data storage path.

    Returns:
        str: Path to XAlign data directory
    """
    try:
        import xalign.file as xfile

        return xfile.get_data_path()
    except ImportError:
        # Fallback to common locations if xalign not available
        possible_paths = [
            os.path.expanduser("~/.xalign/data"),
            os.path.expanduser("~/xalign_data"),
            os.path.join(os.getcwd(), "xalign_data"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Default fallback
        return os.path.expanduser("~/.xalign/data")


def check_index_integrity(data_path: str) -> dict[str, any]:
    """
    Check the integrity of XAlign reference indexes.

    Args:
        data_path: Path to XAlign data directory

    Returns:
        Dict containing integrity check results
    """
    integrity_info = {
        "status": "unknown",
        "indexes_found": [],
        "corrupted_indexes": [],
        "missing_files": [],
        "total_size_gb": 0.0,
        "recommendations": [],
    }

    try:
        if not os.path.exists(data_path):
            integrity_info["status"] = "no_data_dir"
            integrity_info["recommendations"].append(
                f"Create data directory: {data_path}"
            )
            return integrity_info

        # Look for common index file patterns
        # We'll check for common index file extensions in the file loop below

        found_files = []
        total_size = 0

        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size

                # Check for common index file extensions
                if any(
                    file.endswith(ext.replace("*", ""))
                    for ext in [".idx", ".fa", ".fasta", ".gtf", ".gff"]
                ):
                    found_files.append(
                        {
                            "path": file_path,
                            "size_mb": file_size / (1024 * 1024),
                            "modified": os.path.getmtime(file_path),
                        }
                    )

        integrity_info["indexes_found"] = found_files
        integrity_info["total_size_gb"] = total_size / (1024**3)

        # Basic integrity checks
        if len(found_files) == 0:
            integrity_info["status"] = "no_indexes"
            integrity_info["recommendations"].append(
                "No reference indexes found - will be built on first use"
            )
        elif integrity_info["total_size_gb"] < 0.1:
            integrity_info["status"] = "suspicious_size"
            integrity_info["recommendations"].append(
                "Index files seem unusually small - may be corrupted"
            )
        else:
            integrity_info["status"] = "indexes_present"
            integrity_info["recommendations"].append(
                "Reference indexes found and appear normal"
            )

        # Check for zero-byte files (common corruption indicator)
        for file_info in found_files:
            if file_info["size_mb"] == 0:
                integrity_info["corrupted_indexes"].append(file_info["path"])
                integrity_info["recommendations"].append(
                    f"Remove zero-byte file: {file_info['path']}"
                )

        if integrity_info["corrupted_indexes"]:
            integrity_info["status"] = "corruption_detected"

    except Exception as e:
        integrity_info["status"] = "check_failed"
        integrity_info["recommendations"].append(f"Integrity check failed: {e}")

    return integrity_info


def check_disk_space(data_path: str) -> tuple[bool, dict[str, float]]:
    """
    Check available disk space for index building.

    Args:
        data_path: Path to check disk space for

    Returns:
        Tuple of (sufficient_space, space_info_dict)
    """
    try:
        total, used, free = shutil.disk_usage(data_path)

        space_info = {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100,
        }

        # We recommend at least 25GB free for safe index building
        sufficient = space_info["free_gb"] >= 25

        return sufficient, space_info

    except Exception as e:
        logger.error(f"Could not check disk space: {e}")
        return True, {"error": str(e)}


def test_xalign_functionality() -> dict[str, any]:
    """
    Test XAlign basic functionality and connectivity.

    Returns:
        Dict containing test results
    """
    test_results = {
        "import_success": False,
        "data_path_accessible": False,
        "ensembl_connection": False,
        "error_messages": [],
    }

    try:
        # Test xalign import
        import xalign.ensembl as ensembl
        import xalign.file as xfile

        test_results["import_success"] = True

        # Test data path
        data_path = xfile.get_data_path()
        os.makedirs(data_path, exist_ok=True)
        test_results["data_path_accessible"] = True
        test_results["data_path"] = data_path

        # Test Ensembl connection
        try:
            organisms = ensembl.retrieve_ensembl_organisms(release=None)
            if "homo_sapiens" in organisms:
                release = organisms["homo_sapiens"][5]
                test_results["ensembl_connection"] = True
                test_results["ensembl_release"] = release
            else:
                test_results["error_messages"].append(
                    "Ensembl connection successful but 'homo_sapiens' not found"
                )
        except Exception as e:
            test_results["error_messages"].append(f"Ensembl connection failed: {e}")

    except ImportError as e:
        test_results["error_messages"].append(f"XAlign import failed: {e}")
    except Exception as e:
        test_results["error_messages"].append(f"XAlign test failed: {e}")

    return test_results


def generate_index_report(data_path: str | None = None) -> str:
    """
    Generate a comprehensive report on XAlign index status.

    Args:
        data_path: Optional path to XAlign data directory

    Returns:
        Formatted report string
    """
    if data_path is None:
        data_path = get_xalign_data_path()

    report_lines = []
    report_lines.append("🔍 XAlign Index Status Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Basic functionality test
    func_test = test_xalign_functionality()
    report_lines.append("📋 Basic Functionality:")
    report_lines.append(f"   Import: {'✅' if func_test['import_success'] else '❌'}")
    report_lines.append(
        f"   Data Path: {'✅' if func_test['data_path_accessible'] else '❌'}"
    )
    report_lines.append(
        f"   Ensembl: {'✅' if func_test['ensembl_connection'] else '⚠️'}"
    )

    if "data_path" in func_test:
        report_lines.append(f"   Location: {func_test['data_path']}")

    if func_test["error_messages"]:
        report_lines.append("   Errors:")
        for error in func_test["error_messages"]:
            report_lines.append(f"     - {error}")

    report_lines.append("")

    # Disk space check
    space_sufficient, space_info = check_disk_space(data_path)
    report_lines.append("💾 Disk Space:")
    if "error" not in space_info:
        report_lines.append(f"   Free: {space_info['free_gb']:.1f}GB")
        report_lines.append(f"   Total: {space_info['total_gb']:.1f}GB")
        report_lines.append(f"   Usage: {space_info['usage_percent']:.1f}%")
        report_lines.append(
            f"   Status: {'✅ Sufficient' if space_sufficient else '⚠️ Limited'}"
        )
    else:
        report_lines.append(f"   Error: {space_info['error']}")

    report_lines.append("")

    # Index integrity check
    integrity = check_index_integrity(data_path)
    report_lines.append("🗂️  Index Integrity:")
    report_lines.append(f"   Status: {integrity['status']}")
    report_lines.append(f"   Indexes Found: {len(integrity['indexes_found'])}")
    report_lines.append(f"   Total Size: {integrity['total_size_gb']:.2f}GB")

    if integrity["corrupted_indexes"]:
        report_lines.append(f"   ❌ Corrupted: {len(integrity['corrupted_indexes'])}")

    if integrity["indexes_found"]:
        report_lines.append("   Index Files:")
        for idx in integrity["indexes_found"][:5]:  # Show first 5
            report_lines.append(
                f"     - {os.path.basename(idx['path'])} ({idx['size_mb']:.1f}MB)"
            )
        if len(integrity["indexes_found"]) > 5:
            report_lines.append(
                f"     ... and {len(integrity['indexes_found']) - 5} more"
            )

    if integrity["recommendations"]:
        report_lines.append("")
        report_lines.append("💡 Recommendations:")
        for rec in integrity["recommendations"]:
            report_lines.append(f"   - {rec}")

    return "\n".join(report_lines)


def clean_corrupted_indexes(
    data_path: str | None = None, dry_run: bool = True
) -> dict[str, any]:
    """
    Clean corrupted or zero-byte index files.

    Args:
        data_path: Optional path to XAlign data directory
        dry_run: If True, only report what would be deleted

    Returns:
        Dict containing cleanup results
    """
    if data_path is None:
        data_path = get_xalign_data_path()

    cleanup_results = {
        "files_to_remove": [],
        "files_removed": [],
        "errors": [],
        "dry_run": dry_run,
    }

    try:
        integrity = check_index_integrity(data_path)

        # Add zero-byte files to removal list
        cleanup_results["files_to_remove"].extend(integrity["corrupted_indexes"])

        # Add any suspicious files (you can extend this logic)
        for file_info in integrity["indexes_found"]:
            if file_info["size_mb"] < 0.001:  # Less than 1KB
                cleanup_results["files_to_remove"].append(file_info["path"])

        if not dry_run:
            for file_path in cleanup_results["files_to_remove"]:
                try:
                    os.remove(file_path)
                    cleanup_results["files_removed"].append(file_path)
                    logger.info(f"Removed corrupted file: {file_path}")
                except Exception as e:
                    error_msg = f"Failed to remove {file_path}: {e}"
                    cleanup_results["errors"].append(error_msg)
                    logger.error(error_msg)

    except Exception as e:
        cleanup_results["errors"].append(f"Cleanup failed: {e}")

    return cleanup_results


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="XAlign index validation and monitoring"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive report"
    )
    parser.add_argument("--clean", action="store_true", help="Clean corrupted indexes")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run for cleaning (default)"
    )
    parser.add_argument("--data-path", help="Custom XAlign data path")

    args = parser.parse_args()

    if args.report or (not args.clean):
        # Generate and print report
        report = generate_index_report(args.data_path)
        print(report)

    if args.clean:
        print("\n" + "=" * 50)
        print("🧹 Index Cleanup")
        print("=" * 50)

        dry_run = not args.dry_run if args.dry_run is False else True
        cleanup = clean_corrupted_indexes(args.data_path, dry_run=dry_run)

        if cleanup["files_to_remove"]:
            print(f"Files to remove ({len(cleanup['files_to_remove'])}):")
            for file_path in cleanup["files_to_remove"]:
                print(f"  - {file_path}")
        else:
            print("No corrupted files found.")

        if not dry_run and cleanup["files_removed"]:
            print(f"\nRemoved {len(cleanup['files_removed'])} corrupted files.")

        if cleanup["errors"]:
            print("\nErrors during cleanup:")
            for error in cleanup["errors"]:
                print(f"  - {error}")

        if dry_run and cleanup["files_to_remove"]:
            print(
                "\nThis was a dry run. Use --clean without --dry-run to "
                "actually remove files."
            )


if __name__ == "__main__":
    main()
