"""Validate and summarize a profile offline; never executes hooks or cloud calls."""
import argparse
import sys
from pathlib import Path

from compute_maintenance.profiles.schema import DEFAULTS, HOOK_STAGES, load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    args = parser.parse_args()
    try:
        profile = load_profile(str(args.profile.resolve()))
    except (OSError, ValueError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2
    # YAML syntax errors have a distinct exception base; keep diagnostics concise.
    except Exception as exc:
        import yaml
        if not isinstance(exc, yaml.YAMLError):
            raise
        print(f'INVALID YAML: {exc}', file=sys.stderr)
        return 2
    print(f"SCHEMA VALID | {profile['name']} | reboot={profile['reboot']}")
    print('Offline validation only: no commands executed, no cloud connection.')
    settings = dict(DEFAULTS, **profile['settings'])
    print('Settings (before CLI overrides):')
    for key, value in settings.items():
        print(f'  {key}: {value}')
    placeholders = []
    for stage in HOOK_STAGES:
        steps = profile['hooks'][stage]
        print(f'{stage}: {len(steps)} hook(s)')
        for step in steps:
            check = 'yes' if step['check_cmd'] else 'no'
            print(f"  {step['id']} | {step['run']} | {step['timeout']}s | "
                  f"{step['idempotency']} | reconcile check={check}")
            if any('PROFILE_NOT_IMPLEMENTED' in (step.get(key) or '') for key in ('cmd', 'check_cmd')):
                placeholders.append(stage + '/' + step['id'])
    if placeholders:
        print('DRAFT: template placeholders remain: ' + ', '.join(placeholders), file=sys.stderr)
        return 2
    print('Review hook semantics and test in lab before apply; schema validity is not runtime acceptance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
