#!/usr/bin/env python3
"""
Interactive script to fix duplicated token descriptions.
- Automatically finds duplicates
- Uses web search to suggest proper descriptions
- Allows manual review/approval before applying changes
"""

import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path


def load_tokens(filepath):
    """Load token descriptions from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_tokens(tokens, filepath):
    """Save tokens to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def find_duplicates(tokens, min_count=5):
    """Find duplicated descriptions."""
    description_map = defaultdict(list)

    for idx, token_entry in enumerate(tokens):
        desc = token_entry.get('description', '')
        if is_generic_description(desc):
            description_map[desc].append(idx)

    # Filter to only duplicates with min_count
    duplicates = {desc: indices for desc, indices in description_map.items()
                  if len(indices) >= min_count}

    return duplicates


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

    return False


def print_token_info(token_entry, index):
    """Print token information."""
    print(f"\n{'='*80}")
    print(f"Token #{index + 1}")
    print(f"{'='*80}")
    print(f"Token:       {token_entry['token']}")
    print(f"Token ID:    {token_entry['token_id']}")
    print(f"Current:     {token_entry['description']}")
    if token_entry.get('original_description'):
        print(f"Original:    {token_entry['original_description']}")


def get_web_search_suggestion(token):
    """
    Returns a search query that the user should manually search.
    This is a placeholder - actual web search would happen externally.
    """
    return f"Korean medical term {token} meaning in English"


def interactive_fix_token(tokens, idx, search_enabled=False):
    """
    Interactively fix a single token.
    Returns True if changed, False if skipped.
    """
    token_entry = tokens[idx]
    token = token_entry['token']
    current_desc = token_entry['description']

    print_token_info(token_entry, idx)

    # Suggest web search
    search_query = get_web_search_suggestion(token)
    print(f"\nSuggested search: {search_query}")

    print("\nOptions:")
    print("  [1] Enter new description manually")
    print("  [2] Search web and enter result")
    print("  [s] Skip this token")
    print("  [q] Quit and save progress")

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            return 'quit'
        elif choice == 's':
            print("Skipped.")
            return False
        elif choice in ['1', '2']:
            break
        else:
            print("Invalid choice. Try again.")

    # If user chose to search (option 2), remind them
    if choice == '2':
        print(f"\n{'─'*80}")
        print(f"Please search: {search_query}")
        print(f"{'─'*80}")
        print("Waiting for you to search and return with the description...")

    # Get new description
    print("\nEnter new description (or press Enter to skip):")
    new_desc = input("> ").strip()

    if not new_desc:
        print("No description entered. Skipped.")
        return False

    # Confirm change
    print(f"\n{'─'*80}")
    print(f"Token:   {token}")
    print(f"Old:     {current_desc}")
    print(f"New:     {new_desc}")
    print(f"{'─'*80}")

    confirm = input("Apply this change? [y/N]: ").strip().lower()

    if confirm == 'y':
        # Save original description
        if not token_entry.get('original_description'):
            token_entry['original_description'] = current_desc

        token_entry['description'] = new_desc
        token_entry['manually_reviewed'] = True

        print("✓ Applied!")
        return True
    else:
        print("Change cancelled.")
        return False


def batch_fix_duplicates(tokens, duplicates, batch_size=20):
    """
    Fix duplicates in batches.
    Returns number of changes made.
    """
    changes_made = 0
    total_to_fix = sum(len(indices) for indices in duplicates.values())

    print(f"\n{'='*80}")
    print(f"BATCH FIXING DUPLICATES")
    print(f"{'='*80}")
    print(f"Total duplicate tokens to review: {total_to_fix}")
    print(f"Processing in batches of {batch_size}")
    print(f"{'='*80}\n")

    # Flatten all duplicate indices
    all_indices = []
    for desc, indices in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
        for idx in indices:
            all_indices.append((idx, desc))

    # Process in batches
    for batch_start in range(0, len(all_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(all_indices))
        batch = all_indices[batch_start:batch_end]

        print(f"\n{'#'*80}")
        print(f"Batch {batch_start//batch_size + 1}: "
              f"Processing tokens {batch_start + 1}-{batch_end} of {len(all_indices)}")
        print(f"{'#'*80}")

        for i, (idx, desc) in enumerate(batch):
            result = interactive_fix_token(tokens, idx)

            if result == 'quit':
                print(f"\n{'='*80}")
                print(f"Quitting. Made {changes_made} changes so far.")
                print(f"{'='*80}")
                return changes_made

            if result:
                changes_made += 1

        # After each batch, ask if user wants to continue
        if batch_end < len(all_indices):
            print(f"\n{'─'*80}")
            print(f"Batch complete. {changes_made} changes made so far.")
            print(f"Remaining tokens: {len(all_indices) - batch_end}")
            print(f"{'─'*80}")

            cont = input("Continue to next batch? [Y/n]: ").strip().lower()
            if cont == 'n':
                print(f"Stopping. Made {changes_made} changes total.")
                return changes_made

    print(f"\n{'='*80}")
    print(f"All batches complete! Made {changes_made} changes total.")
    print(f"{'='*80}")

    return changes_made


def main():
    parser = argparse.ArgumentParser(
        description="Interactively fix duplicated token descriptions"
    )
    parser.add_argument(
        '--input',
        default='data/tokenizer/reviewed_new_token_description.json',
        help='Input JSON file'
    )
    parser.add_argument(
        '--output',
        help='Output JSON file (default: same as input with _fixed suffix)'
    )
    parser.add_argument(
        '--min-duplicates',
        type=int,
        default=10,
        help='Minimum occurrences to consider as duplicate (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of tokens to process per batch (default: 20)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of original file'
    )

    args = parser.parse_args()

    # Determine output file
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")

    print(f"Loading tokens from {args.input}...")
    tokens = load_tokens(args.input)
    print(f"Loaded {len(tokens)} tokens\n")

    # Create backup if requested
    if args.backup:
        backup_path = args.input + '.backup'
        save_tokens(tokens, backup_path)
        print(f"Created backup: {backup_path}\n")

    # Find duplicates
    print("Finding duplicates...")
    duplicates = find_duplicates(tokens, min_count=args.min_duplicates)

    total_dups = sum(len(indices) for indices in duplicates.values())
    print(f"\nFound {len(duplicates)} duplicate descriptions")
    print(f"Total duplicate tokens: {total_dups}")

    # Show summary
    print("\n" + "─"*80)
    print("TOP DUPLICATES:")
    print("─"*80)
    for desc, indices in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  '{desc}' - {len(indices)} tokens")

    print("\n" + "="*80)
    input("Press Enter to start interactive fixing...")

    # Start batch fixing
    changes_made = batch_fix_duplicates(tokens, duplicates, batch_size=args.batch_size)

    # Save if changes were made
    if changes_made > 0:
        print(f"\nSaving {changes_made} changes to {args.output}...")
        save_tokens(tokens, args.output)
        print(f"✓ Saved!")

        # Show summary of changes
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total tokens processed: {len(tokens)}")
        print(f"Changes made: {changes_made}")
        print(f"Output file: {args.output}")
        print(f"{'='*80}")
    else:
        print("\nNo changes made. File not saved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
