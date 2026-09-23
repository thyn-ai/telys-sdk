# Contributing to the Telys SDK

Thank you for your interest in contributing. This repository holds the
**Python SDK and MCP server** for Telys — embedded, on-device memory &
retrieval for AI assistants — the parts of the product that are open source
and meant to be forked, read, and improved by anyone.

The Telys engine itself (the signed on-device runtime the SDK drives) is
closed and distributed separately — `telys login` fetches it at install time.
Nothing in this repository grants access to it, and nothing you contribute
here can change how much capacity any license is entitled to — that's
enforced entirely on the engine side, independent of this SDK. See
[SECURITY.md](./SECURITY.md) for the trust boundary this implies for
security reports.

## This repository is an automated mirror

Every file here is mirrored from the private `thyn-ai/telys` repository by
an automated sync — nothing is exempt, including these community files. A
pull request that lands here is reviewed and merged normally, and a
maintainer then ports the change upstream; it flows back out here on the
next sync run. You do not need to do anything special — just open the PR
here — but please don't be surprised when the commit that "sticks" arrives
via the sync bot rather than your original commit.

## What you can contribute

| Area | Status | Notes |
|------|--------|-------|
| `packages/telys-sdk/` | ✅ Open | Bug fixes, type fixes, new SDK methods, docs |
| Examples / docs | ✅ Open | Corrections, new guides |
| The on-device runtime | 🔒 Closed | Not in this repository — see above |

## Getting started

```bash
git clone https://github.com/thyn-ai/telys-sdk
cd telys-sdk/packages/telys-sdk
pip install -e .
```

The SDK is a thin client over the signed on-device runtime: `telys login`
fetches a free community license and the runtime, and everything runs
locally thereafter — you do not need any cloud account to work on or use
this code. Note the mirror is source-only: the package's test suite lives
upstream and is not mirrored here (see "automated mirror" above), so PRs
are gated by review and the checks below rather than by a local `pytest`
run in this repo.

## Development workflow

### Branch naming
- `feat/short-description` — new feature
- `fix/short-description` — bug fix
- `docs/short-description` — documentation only

### Commit messages
We follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(sdk): add collection compaction helper
fix(mcp): correct pagination cursor on list_memories
docs(readme): document pipx install path
```

### Pull request checklist
- [ ] The change belongs to the mirrored public surface (see above)
- [ ] New SDK behavior is documented in the package README or docstrings
- [ ] No hardcoded credentials or secrets
- [ ] I understand a maintainer will port the merged change upstream (see
      "automated mirror" above)

All required checks must pass, including on forked-repository pull
requests — CI runs with no secrets and no elevated permissions, so it's
safe to run automatically on every PR.

## Recognizing contributors

This project follows the [all-contributors](https://allcontributors.org)
specification: everyone who contributes — code, docs, bug reports, reviews,
or any other [contribution type](https://allcontributors.org/docs/en/emoji-key) —
is recognized in the [README](./README.md#contributors). Maintainers add
contributors by commenting `@all-contributors please add @user for code`
(replacing `code` with the relevant contribution type) on an issue or pull
request, and the bot opens a pull request updating the contributors table.

## Licensing

By submitting a pull request you agree that your contribution is licensed
under the project's [Apache-2.0 license](./LICENSE) (inbound=outbound,
[GitHub Terms of Service §D.6](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#6-contributions-under-repository-license)).

## Reporting issues

- **Security vulnerabilities** → see [SECURITY.md](./SECURITY.md) (do NOT
  open a public issue)
- **Bugs** → [GitHub Issues](https://github.com/thyn-ai/telys-sdk/issues)
  with the `bug` label
- **Feature requests** → GitHub Issues with the `enhancement` label
- **Questions** → GitHub Issues with the `question` label, or
  https://discord.gg/w8NDsph9an

## Community

- Discord: https://discord.gg/w8NDsph9an
- Docs: https://docs.telys.ai
- Email: community@algenta.ai
