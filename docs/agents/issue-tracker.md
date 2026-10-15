# Issue tracker: GitHub

This repository tracks issues and PRDs in GitHub Issues. Use the `gh` CLI for tracker operations.

## Conventions

- Create issues with `gh issue create`.
- Read issues with `gh issue view <number> --comments`.
- List issues with `gh issue list`; add state or label filters only when needed.
- Comment with `gh issue comment <number> --body "..."`.
- Apply or remove labels with `gh issue edit`.
- Close issues with `gh issue close`.

The repository remote identifies the GitHub project, so run `gh` inside this checkout.

## Skill interpretation

When an engineering skill says to publish to the issue tracker, create a GitHub Issue. When it says to fetch a ticket, read the specified issue and its comments.
