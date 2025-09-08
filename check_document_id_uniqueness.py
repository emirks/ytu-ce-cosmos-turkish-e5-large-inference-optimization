#!/usr/bin/env python3
"""
Script to check if all document IDs are unique in merged_unique_documents.parquet
"""

import pandas as pd
import sys
from pathlib import Path


def check_document_id_uniqueness(parquet_file_path):
    """
    Check if all document IDs are unique in the parquet file.

    Args:
        parquet_file_path (str): Path to the parquet file

    Returns:
        dict: Analysis results
    """
    try:
        print(f"Loading parquet file: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)

        print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Check if documentid column exists
        if "documentid" not in df.columns:
            print("❌ Error: 'documentid' column not found in the dataset")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Basic statistics
        total_rows = len(df)
        total_document_ids = df["documentid"].count()  # Non-null count
        null_document_ids = df["documentid"].isnull().sum()
        unique_document_ids = df["documentid"].nunique()

        print("\n" + "=" * 60)
        print("DOCUMENT ID UNIQUENESS ANALYSIS")
        print("=" * 60)

        print(f"Total rows in dataset: {total_rows:,}")
        print(f"Non-null document IDs: {total_document_ids:,}")
        print(f"Null document IDs: {null_document_ids:,}")
        print(f"Unique document IDs: {unique_document_ids:,}")

        # Check for uniqueness
        if null_document_ids > 0:
            print(f"\n⚠️  WARNING: Found {null_document_ids:,} null document IDs")

        if unique_document_ids == total_document_ids and null_document_ids == 0:
            print("\n✅ ALL DOCUMENT IDs ARE UNIQUE!")
            return {
                "all_unique": True,
                "total_rows": total_rows,
                "unique_ids": unique_document_ids,
                "duplicates": 0,
                "null_ids": null_document_ids,
            }
        else:
            duplicates_count = total_document_ids - unique_document_ids
            print(f"\n❌ DUPLICATE DOCUMENT IDs FOUND!")
            print(f"Number of duplicate document IDs: {duplicates_count:,}")

            # Find and display duplicates
            if duplicates_count > 0:
                print("\nAnalyzing duplicates...")
                duplicate_ids = df[df.duplicated("documentid", keep=False)][
                    "documentid"
                ]
                duplicate_counts = duplicate_ids.value_counts()

                print(f"\nTop 10 most frequent duplicate document IDs:")
                print("-" * 50)
                for doc_id, count in duplicate_counts.head(10).items():
                    print(f"Document ID: {doc_id} → appears {count} times")

                # Show some example duplicate rows
                print(f"\nExample duplicate rows (first 5 sets):")
                print("-" * 50)
                shown_ids = set()
                for doc_id in duplicate_counts.head(5).index:
                    if doc_id not in shown_ids:
                        duplicate_rows = df[df["documentid"] == doc_id]
                        print(
                            f"\nDocument ID '{doc_id}' appears {len(duplicate_rows)} times:"
                        )
                        # Show key columns for these rows
                        key_cols = [
                            "documentid",
                            "itemtype_name",
                            "esasno",
                            "kararno",
                            "karartarihi",
                        ]
                        available_cols = [col for col in key_cols if col in df.columns]
                        print(duplicate_rows[available_cols].to_string(index=False))
                        shown_ids.add(doc_id)

                # Save duplicates to a separate file
                duplicate_rows_all = df[
                    df.duplicated("documentid", keep=False)
                ].sort_values("documentid")
                output_file = "duplicate_document_ids.csv"
                duplicate_rows_all.to_csv(output_file, index=False)
                print(f"\n💾 All duplicate rows saved to: {output_file}")

            return {
                "all_unique": False,
                "total_rows": total_rows,
                "unique_ids": unique_document_ids,
                "duplicates": duplicates_count,
                "null_ids": null_document_ids,
                "duplicate_counts": duplicate_counts if duplicates_count > 0 else None,
            }

    except FileNotFoundError:
        print(f"❌ Error: File '{parquet_file_path}' not found")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None


def main():
    """Main function to run the uniqueness check."""
    parquet_file = "merged_unique_documents.parquet"

    # Check if file exists
    if not Path(parquet_file).exists():
        print(f"❌ Error: File '{parquet_file}' not found in current directory")
        print(
            "Please make sure the file exists and run the script from the correct directory."
        )
        sys.exit(1)

    # Run the analysis
    results = check_document_id_uniqueness(parquet_file)

    if results is None:
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results["all_unique"]:
        print("✅ Document ID uniqueness: PASSED")
        print("🎉 All document IDs are unique!")
    else:
        print("❌ Document ID uniqueness: FAILED")
        print(f"   - Found {results['duplicates']:,} duplicate document IDs")
        if results["null_ids"] > 0:
            print(f"   - Found {results['null_ids']:,} null document IDs")
        print("   - Check 'duplicate_document_ids.csv' for details")

    print(f"\nDataset summary:")
    print(f"   - Total rows: {results['total_rows']:,}")
    print(f"   - Unique document IDs: {results['unique_ids']:,}")
    print(
        f"   - Uniqueness rate: {(results['unique_ids']/results['total_rows']*100):.2f}%"
    )


if __name__ == "__main__":
    main()
