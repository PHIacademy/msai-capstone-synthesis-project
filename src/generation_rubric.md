# Generator Bake-off: Protocol and Rubric

Purpose: decide whether the Capstone 5 limitation (about 1 in 10 usable items) was a
model-size problem or a method problem, using the same targets for every generator.

## Protocol
1. Fix a target list once: about 10 (module, sub-topic, difficulty band) cells, covering all
   five modules, text-only, including at least 2 thin sub-topics. Save it with the results.
2. For each generator in the config, generate 3 candidates per target (about 30 items each).
   Use the same grounding (curriculum text and few-shot examples) for the structured-prompt
   models. The legacy Capstone 5 models use their original header, which is the fair
   reproduction of that baseline.
3. Pool all candidates, strip the generator name, and shuffle with a fixed seed.
4. Score blind, one item at a time, with the rubric below. Keep the unblinding key in a
   separate file until all items are scored.
5. Rate the items yourself as the subject-matter expert. Re-score a random 20% a day later
   and report your own agreement rate as a reliability check.
6. Only after scoring, unblind and compute the metrics.

## Per-item rubric (score each criterion 0 or 1)
| Code | Criterion | Score 1 when |
|---|---|---|
| F | Format | Stem plus exactly four options A-D, no leftover text, no reference to a figure |
| K | Single correct answer | Exactly one option is defensible and it is the one the generator keyed (or the key is absent and one option is clearly correct) |
| D | Distractors | The other three are plausible, mutually distinct, and none is correct or absurd; no option contains another; the key is not obviously longer or more specific |
| C | Curriculum fit | The item belongs to the requested module and tests content in the curriculum guide |
| L | Level | Difficulty is reasonable for HKDSE Paper 1A and not far from the requested band |
| O | Originality | Not a near-copy of a real bank question (check against the bank, not by memory) |

An item is **usable without edits** only if F, K, D, C and O are all 1 (L is reported but not
required, because difficulty is provisional anyway). An item is **usable after light edits** if
exactly one of K or D fails and the fix would take you under two minutes; record the fix type.

## Metrics per generator
- Valid-format rate (automatic parser, also compared with F as scored)
- Usable-without-edits rate, with a Wilson 95% interval
- Usable-after-light-edits rate
- Failure breakdown (counts of K, D, C, O failures)
- Teacher minutes per usable item (time a stopwatch over a sample)
- Gate agreement (computed later, once gate.py exists): the share of your rejections the gate
  also rejects, and the share of your accepted items the gate wrongly rejects

## Comparing generators
- With about 30 items per generator, differences below roughly 15 percentage points are noise;
  say so. Report intervals, not just point rates.
- Use a two-proportion test (or Fisher's exact test for small counts) with Cohen's h for
  effect size, in line with the Capstone 2 method.
- Compare against the Capstone 5 baselines you already have: base 0% valid format,
  LoRA 90% valid format and 1 in 10 usable.

## Limits to state in the paper
- One expert rater; the reliability check reduces but does not remove this.
- Small samples; one run per generator; stochastic outputs.
- Models may have seen HKDSE papers, so originality is checked against the real bank only.
- Usability is a teacher judgement, not a measure of student learning.

## Scoring sheet (CSV columns)
item_id, target_module, target_subtopic, target_band, F, K, D, C, L, O, usable_no_edit,
usable_light_edit, edit_note, minutes, rater_notes
