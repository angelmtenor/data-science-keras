# Git Commit Message Style Guide
Angel Martinez Tenor. 2018-2022

Message Structure:
- type: subject
- body (optional)

## Types

- feat: a new feature
- fix: a bug fix
- doc or docs: changes to documentation
- clean or style : formatting, missing semi colons, etc; no code change
- refactor: refactoring production code
- test: adding tests, refactoring test; no production code change
- setup or chore: updating build tasks, package manager configs, etc; no production code change
- data: adding or updating data files (csv, parquet ...)

## Subject
Subjects should be no greater than 50 characters, should begin with a capital letter and do not end with a period. Use an imperative tone to describe what a commit does, rather than what it did. For example, use change; not changed or changes.

## Examples
- feat: Replace Dense layers by LSTM layers
- docs: Add instructions for executing the open OCR app
- refactor: Move the missing features function to the helper library
- feat: Add a dummy classifier to the helper library
- feat: Automate (stratified) training/test and features/target splits
- fix: Display categorical targets (previously not shown)
- fix: Binary target parsed to float instead of int

## Credits
Adapted from [Udacity Git Commit Message Style Guide](https://udacity.github.io/git-styleguide/)
