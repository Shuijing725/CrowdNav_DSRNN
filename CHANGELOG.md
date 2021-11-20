# Changelog

All notable changes to this project will be documented in this file.

## 2021-11-20
### Changed
- Added comments in config.py and crowd_sim/

### Removed
- Redundant configs that are never used in config.py

## 2021-07-13
### Added
- unicycle example model

### Changed
- Fix an index issue for last_pos in evaluation.py (see issue #7)
- Merged config.py and argument.py
- Save checkpoints by saving only the state_dict instead of the entire model
- In the CrowdSimDict environment, separated the temporal edge and spatial edges into different keys, and applied this change to SRNN network too
- Improved plot.py

### Removed
- argument.py

## 2021-07-01
### Added
- unicycle example model

### Changed
- Minor bug fix

## 2021-03-19
Initial commit
