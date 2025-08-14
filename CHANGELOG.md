# CHANGELOG



## v2.35.0 (2025-08-14)

### Documentation

* docs: Update PULL_REQUEST_TEMPLATE.md (#23) ([`510e70f`](https://github.com/ExpediaGroup/kamae/commit/510e70fb88ae37955b8cca3062e6cfb742f8c9b2))

### Feature

* feat: Add MinMaxScale estimator, transformer &amp; layer (#21)

* feat: Add MinMaxScale estimator, transformer &amp; layer

Adds a min max scaling op in similar vein to the standard scaler

* docs: Add missing warnings and docstrings

* refactor: Align subtract calls

* tests: Add tests for None min/max values

* chore: Align both to math.

* docs: Improve docstrings and typos ([`0aebfd4`](https://github.com/ExpediaGroup/kamae/commit/0aebfd47b6a482ae6b6d1f32de0228e6c3051a30))

### Unknown

* tests: Remove show commands in tests (#24) ([`8005828`](https://github.com/ExpediaGroup/kamae/commit/8005828f574108e200ca0cfc261f5c97fa1fd600))


## v2.34.1 (2025-06-27)

### Fix

* fix: Ensure that pipeline metadata writes use the rdd api (#13)

* fix: Ensure that pipeline metadata writes use the rdd api

* chore: Add license header ([`8d129c5`](https://github.com/ExpediaGroup/kamae/commit/8d129c56f6c35353bc2a748b1a14593aac3a3ec6))


## v2.34.0 (2025-06-11)

### Documentation

* docs: Minor docs additions (#18)

* build: Add license information to toml

* docs: Add pypi version badge ([`7fa6495`](https://github.com/ExpediaGroup/kamae/commit/7fa64950b21ca37602ab1649eeaf6851933c9492))

### Feature

* feat: Add optional output names to subset model output (#15)

If the output name asked for is an input, we do not add an identity layer over the top ([`eaec071`](https://github.com/ExpediaGroup/kamae/commit/eaec071a563f81b7e55843e03fc90f5683f7f316))


## v2.33.2 (2025-06-06)

### Fix

* fix: Fix squeeze during listwise top_k operation. (#19)

* fix: Fix squeeze during listwise top_k operation.

* fix: Added tests for get_top_n.
fix: Moved Normalize layer from utils to layers folder, to fix a circular dependency error.

* chore: Move Normalize back to utils

---------

Co-authored-by: ddonghi &lt;ddonghi@expediagroup.com&gt;
Co-authored-by: George Barrowclough &lt;george.d.b@hotmail.com&gt; ([`c28e4a9`](https://github.com/ExpediaGroup/kamae/commit/c28e4a98ac63b677e83798dfe72ac9daefb07627))


## v2.33.1 (2025-06-04)

### Build

* build: Set up trusted publishing github actions (#16)

Minor renaming of actions alongside creating the proper trusted publishing OICD workflow ([`845d984`](https://github.com/ExpediaGroup/kamae/commit/845d9843e896dbfa5f977b48046704a2bbb39107))

* build: Add read perms back in (#3) ([`81af646`](https://github.com/ExpediaGroup/kamae/commit/81af646747647472848c969c80144f481966b210))

* build: Add id-token write back in ([`82cf518`](https://github.com/ExpediaGroup/kamae/commit/82cf518033bf16b9502b66cc662816867b577fc7))

* build: Update tokens on publish ([`3f4e717`](https://github.com/ExpediaGroup/kamae/commit/3f4e717d2af71d6e550b709173cdf5509ee62592))

* build: Configure workflows (#2)

* build: Use PAT token for releases

* build: Rename repo url with dashes

* build: Add id token write permissions ([`0639630`](https://github.com/ExpediaGroup/kamae/commit/0639630a3a2f5ab5b74f30c7efa01bc04228a2dc))

### Chore

* chore: Remove workflow dispatch from release and publish (#17)

This should not be needed anymore since it is all automated. ([`3718ed1`](https://github.com/ExpediaGroup/kamae/commit/3718ed1413d560eaf9d2b98de6ff95ebcc0aa5e9))

### Documentation

* docs: Minor docs changes (#4)

* docs: Minor readme changes

* docs: Fix file ending of readme

* build: Temp add verbose as true to understand publish errors

* docs: Update team name

* docs: Add CI badge

* chore: Add codeowners file

Use admin team for now ([`b76b3f6`](https://github.com/ExpediaGroup/kamae/commit/b76b3f6e8abbe4386370f964c559bf7ff502eaba))

### Fix

* fix: Renames OneHotLayer to OneHotEncodeLayer (#14)

* fix: Renames OneHotLayer to OneHotEncodeLayer

Keeps an alias with the name OneHotLayer and so is not a breaking change.

* fix: use deprecation suggestion ([`b625133`](https://github.com/ExpediaGroup/kamae/commit/b625133331d92ac7341a57e3f57ef4ffcb00a32d))


## v2.33.0 (2025-04-17)

### Feature

* feat: Small docs change to trigger publish ([`2753bdc`](https://github.com/ExpediaGroup/kamae/commit/2753bdc3dbeff249f2b6d7c28ce41e8269f4b7c3))

* feat: Kamae is open-source! ([`aa4bc04`](https://github.com/ExpediaGroup/kamae/commit/aa4bc048c33991d172428056ad4fc1ce6c378990))
