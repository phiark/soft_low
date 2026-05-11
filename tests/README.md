# Tests

Test taxonomy:

- `unit/`: math and utility invariants
- `integration/`: pipeline composition
- `contract/`: schema and output invariants

Current contract coverage should focus on:

1. output normalization
2. unknown mass derivation
3. `state_content_entropy` calculation
4. analysis record schema stability
5. manifest provenance and stale-resume rejection
6. artifact hygiene gates for archived maintenance
