"""CPU-only public entry point. No model or training-data download."""
import argparse
import json
from pathlib import Path

import numpy as np

from .selection import export_records, select, validate_run, write_json


def main():
    parser = argparse.ArgumentParser(description='ReCAP portable CPU selection experiment')
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select')
    selection.add_argument('--features', type=Path, required=True)
    selection.add_argument('--output', type=Path, required=True)
    selection.add_argument('--budget', type=int, required=True)
    selection.add_argument('--depth', type=int, default=7)
    selection.add_argument('--seed', type=int, default=42)
    validation = commands.add_parser('validate')
    validation.add_argument('--run', type=Path, required=True)
    demo = commands.add_parser('demo')
    demo.add_argument('--output', type=Path, default=Path('outputs/demo'))
    exporter = commands.add_parser('export')
    exporter.add_argument('--pool', type=Path, required=True)
    exporter.add_argument('--run', type=Path, required=True)
    exporter.add_argument('--output', type=Path, required=True)
    exporter.add_argument('--pad-to', type=int, default=128)
    exporter.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'select':
        result = select(args.features, args.output, args.budget, args.depth, args.seed)
    elif args.command == 'validate':
        result = validate_run(args.run)
    elif args.command == 'export':
        result = export_records(args.pool, args.run, args.output, args.pad_to, args.seed)
    else:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error('Demo output directory must be empty')
        args.output.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(981)
        count, dimensions = 32, 9
        turns = 1+np.arange(count) % 3
        owner = np.repeat(np.arange(count), turns)
        np.savez_compressed(args.output/'features.npz',
            image=rng.normal(size=(count, dimensions)).astype(np.float32),
            text=rng.normal(size=(len(owner), dimensions)).astype(np.float32),
            owner=owner, direction=np.zeros(len(owner), dtype=np.int64))
        write_json(args.output/'pool.json', [{'id': f'synthetic-{i}', 'conversations':
            [{'from': 'human', 'value': 'Synthetic demo question'},
             {'from': 'gpt', 'value': str(i)}]} for i in range(count)])
        result = {'synthetic_demo': True,
            'selection': select(args.output/'features.npz', args.output/'run', 8, depth=2),
            'export': export_records(args.output/'pool.json', args.output/'run', args.output/'subset')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
