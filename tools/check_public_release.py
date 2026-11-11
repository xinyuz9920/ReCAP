"""Check the explicitly enumerated public source release (standard library only)."""
import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    '.gitignore', 'README.md', 'SOURCE_PROVENANCE.json', 'THIRD_PARTY_NOTICES.md',
    'pyproject.toml', 'recap/__init__.py', 'recap/__main__.py', 'recap/frontend.py',
    'recap/graph.py', 'recap/greedy.py', 'recap/backend.py', 'recap/selection.py',
    'docs/METHOD.md', 'examples/recorded_protocol.json', 'results/observed_metrics.json',
    'results/cpu_validation.json', 'tests/test_core.py', 'tests/test_portable.py',
    'tools/check_public_release.py', 'PUBLIC_FILES.json', 'PUBLIC_SCAN.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(generate=False):
    # Construct private-infrastructure patterns without embedding the actual
    # project user or network address in this public checker.
    patterns = {
        'server_absolute_path': re.compile('/HDD'+r'\d+/'),
        'private_ipv4': re.compile(r'\b(?:10'+r'\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+\.\d+(?:\.\d+)?\b'),
        'windows_user_path': re.compile(r'[A-Za-z]:[\\/]Users[\\/]', re.I),
        'private_key': re.compile('BEGIN '+r'(?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY'),
        'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github'+r'_pat_[A-Za-z0-9_]{20,})\b'),
        'aws_access_key': re.compile(r'\bAKI'+r'A[A-Z0-9]{16}\b')}
    if generate:
        manifest = {'scope': 'Explicit reviewed public files; generated arrays, data, model weights and paper drafts excluded',
            'files': FILES,
            'sha256_except_self_and_scan': {name: digest(ROOT/name) for name in FILES
                                           if name not in ['PUBLIC_FILES.json', 'PUBLIC_SCAN.json']}}
        (ROOT/'PUBLIC_FILES.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    missing = [name for name in FILES if not (ROOT/name).is_file() and name != 'PUBLIC_SCAN.json']
    present = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
               and not any(part in ['.git', '__pycache__'] or part.endswith('.egg-info')
                           for part in p.relative_to(ROOT).parts)}
    unexpected = sorted(present-set(FILES))
    findings, large_files = [], []
    for name in FILES:
        if not (ROOT/name).exists() or name == 'PUBLIC_SCAN.json':
            continue
        path = ROOT/name
        if path.stat().st_size > 1024*1024:
            large_files.append(name)
        body = path.read_text(encoding='utf-8')
        for label, pattern in patterns.items():
            if pattern.search(body):
                findings.append({'file': name, 'category': label})
    provenance = json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
    core_hashes = {name: digest(ROOT/name) == provenance['source_files'][source]['sha256']
                   for name, source in provenance['verbatim_files'].items()}
    observed = json.loads((ROOT/'results/observed_metrics.json').read_text())
    full = observed['models']['Full']['scores']
    for method in observed['models'].values():
        expected = sum(s/f*100 for s, f in zip(method['scores'], full))/len(full)
        if abs(method['relative_mean']-expected) > 1e-10:
            findings.append({'file': 'results/observed_metrics.json', 'category': 'relative_mean_mismatch'})
    manifest = json.loads((ROOT/'PUBLIC_FILES.json').read_text())
    changed = [name for name, value in manifest['sha256_except_self_and_scan'].items() if digest(ROOT/name) != value]
    report = {'status': 'passed' if not (missing or unexpected or findings or large_files or changed)
               and all(core_hashes.values()) else 'failed',
              'reviewed_files': len(FILES), 'missing': missing, 'unexpected': unexpected,
              'pattern_findings': findings, 'files_over_1MiB': large_files,
              'changed_since_manifest': changed, 'preserved_core_hashes_match': core_hashes,
              'scope': 'Static credential-pattern, private-path, explicit-file and large-file scan; not a guarantee against every possible secret.'}
    if generate:
        (ROOT/'PUBLIC_SCAN.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-manifest', action='store_true')
    args = parser.parse_args()
    raise SystemExit(inspect(args.generate_manifest))
