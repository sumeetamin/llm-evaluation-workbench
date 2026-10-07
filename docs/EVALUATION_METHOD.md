# Evaluation method

## Objective

Catch changes that make a support assistant less safe or less useful on the behaviors its owner has explicitly specified.

## Dataset composition

The included 16 examples are hand-authored, synthetic cases split across policy answers, missing account data, ambiguous requests, unsupported claims and malicious instructions inside retrieved text. They exercise the included graders; they are not sampled from user traffic and do not represent real-world performance.

## Graders

| Check | Rule | Interpretation |
| --- | --- | --- |
| Output present | Non-empty answer | Detects missing model results |
| Required content | Every configured phrase appears | Exact assertion check; paraphrases may fail |
| Forbidden content | No configured phrase appears | Regression guard for selected claims |
| Refusal behavior | Small phrase matcher equals expected boolean | A coarse refusal signal, not policy understanding |
| Required citations | Every expected `[[source_id]]` appears | Verifies citation presence |
| Citation allowlist | Every citation ID exists in supplied context | Detects invented source identifiers |

The case pass rate is the fraction passing every configured check. Per-check rates are shown alongside it so one aggregate score does not conceal a particular failure type. Missing outputs fail a case. Extra output IDs are listed.

## Comparing runs

Runs are graded on the same dataset. The comparison reports the pass-rate delta in percentage points, case regressions and improvements. A score delta alone does not establish statistical significance or prove one model is better. Review changed cases and latency/cost trade-offs.

## Human review

Deterministic checks do not judge semantic correctness, tone, completeness, or whether evidence truly entails each sentence. `evalbench review export` creates a local packet that omits expected phrases, automated grades, model identity, and run metrics. Use the same rubric and case IDs across at least two reviewers; packets can be concatenated as JSONL after review.

`evalbench review agreement` reports raw agreement and unweighted Cohen's kappa for each rubric dimension and reviewer pair. It compares only cases labeled by both members of a pair; `unsure` is an explicit category. The reported mean is the arithmetic mean across defined pairwise kappas, not a multi-rater statistic. Kappa can be undefined when reviewers use only one category and is sensitive to category prevalence. Always inspect shared-case counts, raw agreement, and disagreement cases. Agreement describes consistency, not validity or answer quality; reviewers should discuss disagreements and record adjudicated labels separately.

Before a release decision, review representative passes and failures, calibrate graders against multiple reviewers, and keep a held-out set that was not used to tune the prompt.

