#!/usr/bin/env python3
"""
Simple export/import workflow for fixing duplicates:
1. Export duplicates to text file with token info
2. Manually search web and fill in new descriptions
3. Import the filled file and apply changes
"""

import json
import argparse
import re
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
    return desc_lower in generic_terms or len(desc_lower) < 5


def find_duplicates(tokens, min_count=10):
    """Find duplicated descriptions."""
    description_map = defaultdict(list)

    for idx, token_entry in enumerate(tokens):
        desc = token_entry.get('description', '')
        if is_generic_description(desc):
            description_map[desc].append(idx)

    duplicates = {desc: indices for desc, indices in description_map.items()
                  if len(indices) >= min_count}

    return duplicates


def export_duplicates(tokens, duplicates, output_file):
    """Export duplicates to editable text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DUPLICATE TOKEN DESCRIPTIONS - MANUAL FIX FILE\n")
        f.write("=" * 80 + "\n\n")

        f.write("INSTRUCTIONS:\n")
        f.write("-" * 80 + "\n")
        f.write("1. For each token below, search the web for its proper meaning\n")
        f.write("2. Replace 'NEW_DESCRIPTION:' with the proper description you found\n")
        f.write("3. Optionally add 'SOURCE_URL:' with reference URL\n")
        f.write("4. Save this file\n")
        f.write("5. Run: python script/export_import_duplicates.py --import <this_file>\n")
        f.write("-" * 80 + "\n\n")

        f.write("FORMAT FOR EACH TOKEN:\n")
        f.write("-" * 80 + "\n")
        f.write("TOKEN_ID: 123456\n")
        f.write("TOKEN: 예제\n")
        f.write("CURRENT_DESC: medical term\n")
        f.write("SEARCH_QUERY: Korean medical term 예제 meaning\n")
        f.write("NEW_DESCRIPTION: [FILL THIS - your searched description]\n")
        f.write("SOURCE_URL: [OPTIONAL - reference URL]\n")
        f.write("-" * 80 + "\n\n")

        # Calculate approximate line numbers
        def calc_line_number(idx):
            # JSON format: line 1 = [, then 6 lines per entry
            return 2 + (idx * 6)

        total_tokens = 0

        # Group by duplicate description
        for desc, indices in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"DUPLICATE GROUP: '{desc}' ({len(indices)} tokens)\n")
            f.write("=" * 80 + "\n\n")

            for idx in indices:
                token_entry = tokens[idx]
                token = token_entry['token']
                token_id = token_entry['token_id']
                line_num = calc_line_number(idx)

                f.write("-" * 80 + "\n")
                f.write(f"TOKEN_ID: {token_id}\n")
                f.write(f"TOKEN: {token}\n")
                f.write(f"CURRENT_DESC: {desc}\n")
                f.write(f"LINE_NUMBER: ~{line_num}\n")
                f.write(f"SEARCH_QUERY: Korean medical term {token} meaning\n")
                f.write(f"NEW_DESCRIPTION: \n")
                f.write(f"SOURCE_URL: \n")
                f.write("\n")

                total_tokens += 1

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"TOTAL TOKENS TO FIX: {total_tokens}\n")
        f.write("=" * 80 + "\n")

    print(f"Exported {total_tokens} tokens to {output_file}")


def import_duplicates(tokens, input_file):
    """Import manually filled descriptions."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse each token block
    pattern = r'-{80}\nTOKEN_ID: (\d+)\nTOKEN: (.+?)\nCURRENT_DESC: (.+?)\nLINE_NUMBER: (.+?)\nSEARCH_QUERY: (.+?)\nNEW_DESCRIPTION: (.*?)\nSOURCE_URL: (.*?)\n'

    matches = re.findall(pattern, content, re.MULTILINE)

    updates = []
    for match in matches:
        token_id, token, current_desc, line_num, search_query, new_desc, source_url = match
        token_id = int(token_id)
        new_desc = new_desc.strip()
        source_url = source_url.strip()

        if new_desc and new_desc != current_desc:
            updates.append({
                'token_id': token_id,
                'token': token,
                'old_desc': current_desc,
                'new_desc': new_desc,
                'source_url': source_url
            })

    # Apply updates
    if not updates:
        print("No updates found in file. Make sure NEW_DESCRIPTION fields are filled.")
        return 0

    print(f"\nFound {len(updates)} updates to apply\n")

    # Show preview
    print("=" * 80)
    print("PREVIEW OF CHANGES")
    print("=" * 80)

    for i, update in enumerate(updates[:10], 1):
        print(f"\n{i}. Token: {update['token']} (ID: {update['token_id']})")
        print(f"   Old: {update['old_desc']}")
        print(f"   New: {update['new_desc']}")
        if update['source_url']:
            print(f"   Source: {update['source_url']}")

    if len(updates) > 10:
        print(f"\n... and {len(updates) - 10} more")

    print("\n" + "=" * 80)
    confirm = input(f"\nApply {len(updates)} changes? [y/N]: ").strip().lower()

    if confirm != 'y':
        print("Cancelled.")
        return 0

    # Apply changes
    updated_count = 0
    for token_entry in tokens:
        for update in updates:
            if token_entry['token_id'] == update['token_id']:
                # Save original
                if not token_entry.get('original_description'):
                    token_entry['original_description'] = token_entry['description']

                token_entry['description'] = update['new_desc']
                token_entry['manually_reviewed'] = True

                if update['source_url']:
                    token_entry['source_url'] = update['source_url']

                updated_count += 1
                break

    print(f"\n✓ Applied {updated_count} changes!")
    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Export/import duplicates for manual fixing"
    )
    parser.add_argument(
        '--input-json',
        default='data/tokenizer/reviewed_new_token_description.json',
        help='Input JSON file with tokens'
    )
    parser.add_argument(
        '--output-json',
        help='Output JSON file (default: input with _fixed suffix)'
    )
    parser.add_argument(
        '--export',
        help='Export duplicates to this text file for manual editing'
    )
    parser.add_argument(
        '--import-file',
        help='Import manually edited file and apply changes'
    )
    parser.add_argument(
        '--min-duplicates',
        type=int,
        default=10,
        help='Minimum occurrences to export (default: 10)'
    )

    args = parser.parse_args()

    # Determine output JSON file
    if not args.output_json:
        input_path = Path(args.input_json)
        args.output_json = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")

    print(f"Loading tokens from {args.input_json}...")
    tokens = load_tokens(args.input_json)
    print(f"Loaded {len(tokens)} tokens\n")

    # Export mode
    if args.export:
        print("Finding duplicates...")
        duplicates = find_duplicates(tokens, min_count=args.min_duplicates)

        total_dups = sum(len(indices) for indices in duplicates.values())
        print(f"Found {len(duplicates)} duplicate descriptions ({total_dups} tokens)\n")

        export_duplicates(tokens, duplicates, args.export)

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print(f"1. Open {args.export}")
        print(f"2. For each token, search web and fill NEW_DESCRIPTION")
        print(f"3. Save the file")
        print(f"4. Run: python {__file__} --import-file {args.export}")
        print("=" * 80)

    # Import mode
    elif args.import_file:
        updated = import_duplicates(tokens, args.import_file)

        if updated > 0:
            print(f"\nSaving to {args.output_json}...")
            save_tokens(tokens, args.output_json)
            print(f"✓ Saved!\n")

            print("=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Input:   {args.input_json}")
            print(f"Output:  {args.output_json}")
            print(f"Updated: {updated} tokens")
            print("=" * 80)

    else:
        parser.print_help()
        print("\n" + "=" * 80)
        print("QUICK START:")
        print("=" * 80)
        print(f"# Export duplicates for manual fixing:")
        print(f"python {__file__} --export data/tokenizer/duplicates_manual_fix.txt")
        print()
        print(f"# Import after manual editing:")
        print(f"python {__file__} --import-file data/tokenizer/duplicates_manual_fix.txt")
        print("=" * 80)


if __name__ == "__main__":
    main()
