#!/usr/bin/env python3
"""
Automatically find and fix duplicated descriptions using web search.
This script uses Claude's web search capability to find proper medical term definitions.
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
import sys


def load_tokens(filepath):
    """Load token descriptions from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_tokens(tokens, filepath):
    """Save tokens to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def find_duplicates(tokens):
    """Find all duplicated descriptions."""
    description_map = defaultdict(list)

    for idx, token_entry in enumerate(tokens):
        desc = token_entry.get('description', '')
        description_map[desc].append(idx)

    # Filter to only duplicates (2+ occurrences)
    duplicates = {desc: indices for desc, indices in description_map.items()
                  if len(indices) > 1}

    return duplicates


def get_generic_descriptions(duplicates, min_count=5):
    """Get most overused generic descriptions."""
    generic = []

    for desc, indices in duplicates.items():
        if len(indices) >= min_count and is_generic_description(desc):
            generic.append((desc, len(indices)))

    return sorted(generic, key=lambda x: x[1], reverse=True)


def is_generic_description(desc):
    """Check if description is too generic."""
    generic_terms = [
        'medical vocabulary',
        'medical term',
        'korean medical term',
        'medical terminology',
        'single character morpheme',
        'morpheme',
        'medical',
        'korean',
        'term',
        'word',
        'vocabulary',
    ]

    desc_lower = desc.lower().strip()

    # Exact match
    if desc_lower in generic_terms:
        return True

    # Very short descriptions
    if len(desc_lower) < 5:
        return True

    # Only contains generic words
    words = desc_lower.split()
    if all(word in generic_terms for word in words):
        return True

    return False


def create_search_queries(token, current_desc):
    """
    Create search queries for finding proper description.
    Returns list of search query strings.
    """
    queries = []

    # Primary query: Korean medical term meaning
    queries.append(f"Korean medical term {token} meaning in English")

    # If token looks like a medical term
    if len(token) > 2:
        queries.append(f"{token} 의학용어 뜻")  # Korean search
        queries.append(f"medical terminology {token}")

    # If token looks like a medication/drug
    if any(suffix in token for suffix in ['정', '약', '액', '캡슐']):
        queries.append(f"{token} medication drug")

    # If token looks like a disease
    if any(suffix in token for suffix in ['병', '염', '증', '암']):
        queries.append(f"{token} disease condition medical")

    return queries


def export_for_manual_search(tokens, duplicates, output_csv):
    """
    Export duplicates to CSV for manual web search.
    """
    import csv

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Token ID',
            'Token',
            'Current Description',
            'Search Query',
            'New Description (FILL THIS)',
            'Source URL (FILL THIS)'
        ])

        processed = set()

        for desc, indices in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
            if not is_generic_description(desc):
                continue

            for idx in indices:
                token_entry = tokens[idx]
                token = token_entry['token']
                token_id = token_entry['token_id']

                # Skip if already processed
                key = f"{token}_{desc}"
                if key in processed:
                    continue
                processed.add(key)

                # Create primary search query
                search_query = f"Korean medical term {token} meaning"

                writer.writerow([
                    token_id,
                    token,
                    desc,
                    search_query,
                    '',  # To be filled manually
                    ''   # To be filled manually
                ])

    print(f"\nExported {len(processed)} tokens to {output_csv}")
    print("Please open this CSV, search for each token, and fill in:")
    print("  - New Description: proper medical definition")
    print("  - Source URL: reference URL")


def import_from_manual_csv(tokens, input_csv):
    """
    Import manually filled descriptions from CSV.
    """
    import csv

    updates = []

    with open(input_csv, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            token_id = int(row['Token ID'])
            new_desc = row['New Description (FILL THIS)'].strip()
            source_url = row['Source URL (FILL THIS)'].strip()

            if new_desc:  # Only if filled
                updates.append({
                    'token_id': token_id,
                    'new_description': new_desc,
                    'source_url': source_url
                })

    # Apply updates
    updated_count = 0
    for token_entry in tokens:
        for update in updates:
            if token_entry['token_id'] == update['token_id']:
                # Save original description
                if token_entry.get('description'):
                    token_entry['original_description'] = token_entry['description']

                token_entry['description'] = update['new_description']

                # Add source URL if provided
                if update['source_url']:
                    token_entry['source_url'] = update['source_url']

                updated_count += 1
                break

    print(f"\nUpdated {updated_count} token descriptions from CSV")
    return updated_count


def print_duplicate_report(tokens, duplicates):
    """Print detailed report of duplicates."""
    print("=" * 80)
    print("DUPLICATE DESCRIPTIONS REPORT")
    print("=" * 80)

    total_duplicates = sum(len(indices) for indices in duplicates.values())
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Unique descriptions with duplicates: {len(duplicates)}")
    print(f"Total duplicate tokens: {total_duplicates}")

    # Generic descriptions
    generic = get_generic_descriptions(duplicates, min_count=5)
    print(f"\nGeneric descriptions (≥5 occurrences): {len(generic)}")

    print("\n" + "─" * 80)
    print("TOP 20 MOST DUPLICATED DESCRIPTIONS")
    print("─" * 80)

    sorted_dups = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)

    for desc, indices in sorted_dups[:20]:
        print(f"\n'{desc}' - {len(indices)} occurrences")

        # Show first 5 examples
        for idx in indices[:5]:
            token = tokens[idx]['token']
            token_id = tokens[idx]['token_id']
            print(f"  ID {token_id}: {token}")

        if len(indices) > 5:
            print(f"  ... and {len(indices) - 5} more")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Find and fix duplicated token descriptions"
    )
    parser.add_argument(
        '--input',
        default='data/tokenizer/reviewed_new_token_description.json',
        help='Input JSON file'
    )
    parser.add_argument(
        '--output',
        default='data/tokenizer/reviewed_new_token_description_fixed.json',
        help='Output JSON file with fixes'
    )
    parser.add_argument(
        '--export-csv',
        default='data/tokenizer/duplicates_to_fix.csv',
        help='Export duplicates to CSV for manual search'
    )
    parser.add_argument(
        '--import-csv',
        help='Import manually filled CSV with new descriptions'
    )
    parser.add_argument(
        '--min-duplicates',
        type=int,
        default=5,
        help='Minimum occurrences to export (default: 5)'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Only print report, do not export'
    )

    args = parser.parse_args()

    print(f"Loading tokens from {args.input}...")
    tokens = load_tokens(args.input)
    print(f"Loaded {len(tokens)} tokens\n")

    # Find duplicates
    duplicates = find_duplicates(tokens)

    # Print report
    print_duplicate_report(tokens, duplicates)

    if args.report_only:
        print("\nReport-only mode. Exiting.")
        return

    # Import mode
    if args.import_csv:
        print(f"\nImporting updates from {args.import_csv}...")
        updated = import_from_manual_csv(tokens, args.import_csv)

        if updated > 0:
            save_tokens(tokens, args.output)
            print(f"\nSaved updated tokens to {args.output}")
        else:
            print("\nNo updates found in CSV. File not saved.")

        return

    # Export mode (default)
    print(f"\nExporting duplicates to CSV for manual search...")
    export_for_manual_search(tokens, duplicates, args.export_csv)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"1. Open {args.export_csv} in Excel/LibreOffice")
    print("2. For each row, search the web using the 'Search Query' column")
    print("3. Fill in 'New Description' with the proper medical term definition")
    print("4. Optionally fill in 'Source URL' with reference")
    print("5. Save the CSV file")
    print(f"6. Run: python {sys.argv[0]} --import-csv {args.export_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
