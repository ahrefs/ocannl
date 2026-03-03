Consider a hard limit to be 12 pages -- let's aim to have the best paper possible within 12 pages, later we can reconsider the venue or split the paper into 2 page extended abstract and the rest as appendix.

# Follow-up: Solution for gh-ocannl-299

This is a follow-up task based on feedback from PR #430.
PR: https://github.com/ahrefs/ocannl/pull/430

## Original PR Description

Solution from coder for feature: gh-ocannl-299

## PR Comments and Reviews

The comments below (most recent last) describe what needs to be done.
Focus especially on the last comment(s) for the actionable task.

### Comment by claude (2026-03-02T11:40:54Z)

**Claude finished @lukstafi's task** —— [View job](https://github.com/ahrefs/ocannl/actions/runs/22574379821)

---
### PR Review Complete

#### Todo List:
- [x] Read and analyze changed files
- [x] Understand the context of changes
- [x] Review for code quality and best practices
- [x] Check for potential bugs or issues
- [x] Assess performance considerations
- [x] Evaluate security concerns
- [x] Analyze test coverage impact
- [x] Provide constructive feedback

### Code Review Summary

This is a **clean and well-executed maintenance PR** that demonstrates good development practices. The changes are minimal but important for repository health.

### Changes Analysis

#### 1. **Documentation Enhancement** (`AGENTS_STAGING.md`)
**Added valuable development learnings:**
- Documents the pattern of checking git history before implementing new fixes
- Provides specific guidance on using `git log -- <relevant-files>` and `dune build @runtest-<test_name>` 
- Records known test failure (`test_max_pool2d`) to prevent confusion

**Quality:** ✅ **Excellent** - This follows the repository's pattern of collecting agent learnings for future curation into main documentation.

#### 2. **Test Cleanup** (`test_concat_graph.expected`)
**Removed stale error output:**
- Eliminated 6 lines of outdated "Known Limitation" error message
- The underlying bug was already fixed in commit f90d41bf, but the expected file wasn't updated
- Now the test properly shows successful three-way concatenation with full backpropagation

**Quality:** ✅ **Correct** - This is the proper way to maintain test expected files after bug fixes.

### Code Quality Assessment

**✅ Strengths:**
- **Minimal, targeted changes** - Only touches what's necessary
- **Historical awareness** - Author checked git history before making changes (following their own documented best practice)
- **Proper test maintenance** - Updates expected output to match current behavior
- **Good commit messages** - Clear description of what was done and why

**✅ Best Practices Followed:**
- Uses repository's established documentation patterns
- Maintains test hygiene by removing stale outputs
- Includes helpful context in commit messages

### Technical Assessment

**🔧 Performance:** No impact - purely maintenance changes

**🔒 Security:** No security implications

**🧪 Test Coverage:** 
- **Improved** - Test now properly validates three-way concatenation backpropagation
- Removes false negative that was masking actual functionality
- Test coverage is actually enhanced by cleaning up the expected output

**🐛 Bug Analysis:** 
- No bugs introduced
- Actually **fixes** a test maintenance issue
- The actual concatenation bug was already resolved in previous commits

### Verification Notes

From git history analysis:
- Commit f90d41bf: "Fix concat lowering for unit dims" - implemented the actual fix
- Commit aac59bc: Cleaned up the stale expected file (previous round)  
- This PR: Final cleanup and documentation of lessons learned

The test file `test_concat_graph.ml` shows comprehensive testing of:
- 2-way concatenation with backprop
- 3-way concatenation with backprop (now working correctly)
- 3-way concatenation with unit dimensions

### Recommendation

**✅ APPROVE** - This PR demonstrates excellent maintenance practices and should be merged. It:
- Removes technical debt (stale test output)
- Documents valuable development learnings
- Follows repository conventions
- Has zero risk of introducing regressions

The changes are conservative, well-documented, and improve the codebase's maintainability.

### Comment by lukstafi (2026-03-02T15:26:50Z)

## Refactoring Suggestions

*Post-merge retrospective: what would we do differently if starting from scratch?*

# Refactoring Suggestions

## If starting from scratch, I would...

### 1. Investigate the expected file provenance first

The entire task turned out to be removing 6 stale lines from `test_concat_graph.expected`. The actual code fix (commit `f90d41bf`) was already merged to master. I spent time reading through `assignments.ml` concat lowering logic, tracing the `Block` and `Rev_sides` paths, and understanding `allow_by_concat` filtering — all useful for understanding the codebase but unnecessary for the actual change. A faster approach: run `dune build @runtest-test_concat_graph` immediately to see the diff, then check `git log -- test/operations/test_concat_graph.ml` to understand why the expected file was stale.

### 2. The stale expected file should have been caught by the fix commit

The commit `f90d41bf` ("Fix concat lowering for unit dims") modified `test_concat_graph.ml` to add the unit-dim test but left the "Known Limitation" section in the expected file. This means either:
- The expected file was updated separately and a test section was later removed without updating expectations
- Or `dune promote` wasn't run after the test changes

A good practice for future agents: after any test file modification, always run `dune build @runtest-<test_name>` and `dune promote` if needed, rather than manually editing expected files.

### 3. Consider adding a regression test for nested 3-way concat

The current test covers `"a; b; c => a^b^c"` with dims 2,1,2. It would be valuable to add a test for nested concatenation (concatenating results of concatenations) and for the case where the first or last component has dim 1 (not just the middle one), since the `concat_offset_for` computation and `allow_by_concat` filtering interact differently at boundary positions.

### 4. The `allow_by_concat` pattern is duplicated

In `assignments.ml`, the `allow_by_concat` logic (extract concat symbols from `project_lhs`, filter RHS blocks by matching symbol in `block_iters`) is duplicated between `loop_accum` (Block path, ~line 375) and `loop_accum_rev` (Rev_sides path, ~line 560). These could share a helper function like `active_concat_index ~projections ~block_iters ~num_rhses` to reduce duplication and ensure both paths stay in sync.

