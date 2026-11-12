# Profile workspace

Template for writing your own `rolling-maint` maintenance profile. Copy `template/` and follow
[docs/profiles.md](../docs/profiles.md) ([Tiếng Việt](../docs/profiles.vi.md)).

```bash
cp -R profile-workspace/template profile-workspace/my-maintenance
python tools/profile_check.py profile-workspace/my-maintenance/profile.yaml   # offline schema check
```
