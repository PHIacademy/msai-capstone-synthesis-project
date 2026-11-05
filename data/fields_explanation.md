**Document Version:** 1.7
**Date:** 2026-10-23
**Owner:** Stanley Lo

**Version History:**

| Version | Date | Summary of Changes |
| --- | --- | --- |
| 1.0 | 2026-10-05 | Initial release (13 fields). |
| 1.1 | 2026-10-05 | Renamed Fields 2.9–2.11 from `image_filename` / `image_type` / `image_context` to `qn_image_filename` / `qn_image_type` / `qn_image_context` to disambiguate from the new option-image family. Added Fields 2.15–2.17 (`ans_image_filename`, `ans_image_type`, `ans_image_context`) for graphical content embedded within the answer-option block. |
| 1.2 | 2026-10-05 | Documented the three answer-option modalities — `text_only`, `image_only`, `image_plus_text` — inferred from the joint contents of `answer_options` and `ans_image_filename`. The literal sentinel `[IMAGE]` is now an admissible value for `answer_options`, indicating an image-only option block (e.g., HKDSE 2015 Paper 1A Q18). |
| 1.3 | 2026-10-05 | Removed Fields 2.16 (`ans_image_type`) and 2.17 (`ans_image_context`). Answer-option images are now described solely by `ans_image_filename` (Field 2.15); no per-image type or context metadata is retained for the option-block family. |
| 1.4 | 2026-10-05 | Renamed Field 2.7 from `qn_class_id` to `qn_id` (display name "Question Classification ID" → "Question ID"). The field encodes only the subject, paper year, and question number — it carries no classification information, so the former name was misleading. The value format `ICT_{YYYY}_paper1a_q{NN}` is unchanged. |
| 1.5 | 2026-10-05 | Merge-grain redesign. `qn_id` is now the sole primary key (one row per question). `qn_type_id` becomes a semicolon-delimited list (ascending, de-duplicated) with the reserved token `unclassified` for untagged questions. `correct_answer` gains the reserved token `*` for withdrawn/cancelled questions (e.g. 2018 Paper 1A Q40). |
| 1.6 | 2026-10-10 | Added Fields 2.16 (`cluster`) and 2.17 (`cluster_desc`) for "cluster questions" — questions that share a common context (a passage, diagram, or algorithm) with one or more sibling questions. The shared context is now preserved verbatim in `cluster_desc` instead of being dropped. |
| 1.7 | 2026-10-23 | Category names (`cat_name`) aligned to the official module titles in the Curriculum and Assessment Guide (Information and Communication Technology, S4–6): A–E. Typos in the classification workbook corrected ("Infromation", "Ouput"). |

---

# Data Field Specification Documentation

## 1. Overview

This document defines the canonical schema for the structured question dataset. Fields 2.1–2.13 form the core classification/content schema; Field 2.14 (`cohort_pct`) is an supplementary column populated from the Stage 2 answer-key OCR outputs when the underlying printed answer sheet reports cohort performance; Field 2.15 (`ans_image_filename`) records graphical assets embedded within the answer-option block; Fields 2.16 (`cluster`) and 2.17 (`cluster_desc`) capture the shared context of "cluster questions". All fields described below are mandatory unless otherwise specified. Each field entry comprises:

- **Field Name**: Official human-readable identifier
- **Internal Key**: Machine-readable key used in data serialisation
- **Format**: Structural specification of admissible values
- **Constraints**: Enumerated restrictions, invariants, or validation rules
- **Example(s)**: Concrete illustrations of valid values

**Reserved tokens.** The schema reserves three literal values that never collide with ordinary data, and are interpreted specially by downstream consumers: `unclassified` (in `qn_type_id`, §2.6, for a question not tagged under any type), `[IMAGE]` (in `answer_options`, §2.12, for image-only answer options), and `*` (in `correct_answer`, §2.13, for a withdrawn/cancelled question).

---

## 2. Field Definitions

### 2.1 Core Type

| Property | Value |
|---|---|
| Internal Key | `core_type` |
| Format | Single alphanumeric character, no leading or trailing whitespace or symbolic characters |
| Constraints | Identifies the curriculum core module to which the question belongs |

**Example:**

```
b
```

- `b` — Core B

---

### 2.2 Category ID

| Property | Value |
|---|---|
| Internal Key | `cat_id` |
| Format | Two lowercase alphabetic characters |
| Constraints | Value must be one of the predefined category codes: `ip`, `cs`, `ia`, `cp`, `si` |

**Valid Category Code Mapping:**

| Code | Category Name |
|---|---|
| `ip` | Information Processing |
| `cs` | Computer System Fundamentals |
| `ia` | Internet and Its Applications |
| `cp` | Computational Thinking and Programming |
| `si` | Social Implications |

(Names follow the module titles A–E of the HKEAA/EDB ICT Curriculum and Assessment Guide.)

**Example:**

```
ip
```

- `ip` — Category: Information Processing

---

### 2.3 Category Name

| Property | Value |
|---|---|
| Internal Key | `cat_name` |
| Format | Free-form descriptive string (UTF-8) |
| Constraints | Must correspond exactly to the full name associated with `cat_id` |

**Example:**

```
Information Processing
```

---

### 2.4 Subcategory ID

| Property | Value |
|---|---|
| Internal Key | `subcat_id` |
| Format | Two-digit zero-padded decimal string |
| Constraints | Range: `00`–`99`; uniquely identifies a subcategory within a category |

**Example:**

```
01
```

- `01` — Subcategory 01: Introduction to Information Processing

---

### 2.5 Subcategory Name

| Property | Value |
|---|---|
| Internal Key | `subcat_name` |
| Format | Free-form descriptive string (UTF-8) |
| Constraints | Must correspond exactly to the full name associated with `subcat_id` within the parent category |

**Example:**

```
Introduction to Information Processing
```

---

### 2.6 Question Type ID

| Property | Value |
|---|---|
| Internal Key | `qn_type_id` |
| Format | Semicolon-delimited list of one or more composite strings, each of the form `<core_type>_<cat_id>_<subcat_id>`; alternatively the single reserved token `unclassified` |
| Constraints | Each list item must conform to the format and constraints specified for its respective component field (see Sections 2.1, 2.2, 2.4). Items are stored in ascending lexicographic order and de-duplicated. A question that has not been classified under any type carries the single reserved token `unclassified`. |

**Composite Structure:**

| Segment | Source Field | Format |
|---|---|---|
| `<core_type>` | Core Type | 1 character |
| `<cat_id>` | Category ID | 2 characters |
| `<subcat_id>` | Subcategory ID | 2 digits |

**Examples:**

```
a_ip_01
```

- Single type: `a_ip_01`
  - `a` — Core A
  - `ip` — Category IP (Information Processing)
  - `01` — Subcategory ID 01 (Introduction to Information Processing)

```
b_cs_02;c_ia_03
```

- Multiple types (ascending, de-duplicated): the question belongs to both `b_cs_02` and `c_ia_03`

```
unclassified
```

- Unclassified: the question is not tagged under any type

---

### 2.7 Question ID

| Property | Value |
|---|---|
| Internal Key | `qn_id` |
| Format | Composite string with underscore delimiter: `<subject>_<year>_<paper>_<question>` |
| Constraints | Each component must conform to the format and constraints defined below. `qn_id` is the primary key of the dataset — each value is globally unique and corresponds to exactly one row in `final_dataset.csv`. |

**Composite Structure:**

| Segment | Format | Constraints |
|---|---|---|
| `<subject>` | 3 uppercase alphabetic characters | Project-wide constant: `ICT` (Information and Communication Technology) |
| `<year>` | 4-digit decimal string | Represents the examination year |
| `<paper>` | Literal `paper` followed by up to 5 alphanumeric characters | Project-wide convention: `paper1a` |
| `<question>` | Literal `q` followed by 2 zero-padded digits | Identifies the question number within the paper |

**Example:**

```
ICT_2021_paper1a_q01
```

- `ICT` — Subject identifier
- `2021` — Examination year (2021 past paper)
- `paper1a` — Paper 1A
- `q01` — Question number 1

---

### 2.8 Question Text

| Property | Value |
|---|---|
| Internal Key | `question_text` |
| Format | UTF-8 string; may contain embedded image references |
| Constraints | Preserves original typography, layout semantics, and internal image hyperlinks as rendered in the source examination paper |

---

### 2.9 Question-Body Image Filename

| Property | Value |
|---|---|
| Internal Key | `qn_image_filename` |
| Format | Semicolon-delimited list of URI-formatted strings (hyperlinks or relative/absolute file paths); nullable / empty value permitted |
| Constraints | References one or more graphical assets embedded within the **question body** of `question_text` (i.e., between the question stem and the `A./B./C./D.` answer block). Each list item corresponds 1:1 to an entry in the `elements` array of the per-year crop-manifest JSON, in manifest order, **excluding** any element whose `name` ends with the literal suffix `_options` (which is dispatched to `ans_image_filename` — see §2.15). For questions that contain **no** embedded question-body image, this field **must** be left empty (blank / NA). Cardinality — the number of semicolon-separated entries — **must** exactly match the cardinality of `qn_image_type` and `qn_image_context` for the same row. |

> **Note on the rename:** This field was previously named `image_filename`; it was renamed to `qn_image_filename` in v1.1 of this document to disambiguate from the new `ans_image_filename` (Field 2.15) and to make the scope of the field — question body only — explicit at the call site.

---

### 2.10 Question-Body Image Type

| Property | Value |
|---|---|
| Internal Key | `qn_image_type` |
| Format | Semicolon-delimited list of lowercase string literals; nullable / empty value permitted |
| Constraints | Each list item is derived from the `kind` attribute of the corresponding element in the `elements` array of the per-year JSON metadata file (as generated by `scripts/crop_elements.py`). If the `kind` attribute is absent on any given element, that entry **defaults to** `"figure"` rather than being treated as missing data. For questions with no embedded question-body image, this field **must** be left empty (blank / NA). Cardinality **must** exactly match `qn_image_filename` and `qn_image_context` for the same row. Elements whose `name` ends in the literal suffix `_options` are excluded from this field; see §2.15. |

> **Note on the rename:** Previously `image_type`; renamed to `qn_image_type` in v1.1.

**Valid Image Type Enumeration (applies to each semicolon-separated entry):**

| Value | Detection Strategy | Intended Use Case |
|---|---|---|
| `table` | Outermost border lines; falls back to content bounding box with `FALLBACK` flag | Tables, framed boxes, dialog windows, and bounded UI panels |
| `figure` | Content bounding box (ink extent within the container) | Charts, diagrams, flowcharts, composite multi-part figures, photographs, barcodes |
| `code` | Content bounding box (identical algorithm to `figure`) | Source code and pseudocode listings; a fenced ```text block should be appended beneath the image for textual representation |

---

### 2.11 Question-Body Image Context

| Property | Value |
|---|---|
| Internal Key | `qn_image_context` |
| Format | Semicolon-delimited list of lowercase alphanumeric strings (underscores permitted); nullable / empty value permitted |
| Constraints | Each list item is extracted as the descriptive suffix of the `name` field of the corresponding element in the `elements` array of the per-year JSON metadata file — specifically, the segment following the `qNN_` prefix in the element name (e.g., for element name `2025DSE1A_q27_algorithm`, the entry is `algorithm`). For questions with no embedded question-body image, this field **must** be left empty (blank / NA). Cardinality **must** exactly match `qn_image_filename` and `qn_image_type` for the same row. Elements whose `name` ends in the literal suffix `_options` are excluded from this field; see §2.15. |

> **Note on the rename:** Previously `image_context`; renamed to `qn_image_context` in v1.1.

**Observed Question-Body Context Vocabulary (42 values identified across the 2016–2025 dataset; non-exhaustive, illustrative only):**

`algorithm`, `algorithms`, `ascii`, `bar_qr`, `box`, `browser`, `calendar`, `captcha`, `charts`, `cookie`, `diagram`, `dialog`, `email`, `ergonomic`, `eticket`, `files`, `flowchart`, `flowcharts`, `folders`, `form`, `grid`, `information_system`, `ipo`, `methods`, `network`, `options`, `password`, `pivot`, `pseudocode`, `registration`, `remote`, `setting`, `seven_segment`, `slide`, `specs`, `sql`, `statement`, `student`, `table`, `toc`, `versions`, `webpage`

> **Note:** The vocabulary listed above is not a closed or restricted set. The `qn_image_context` field accepts any valid lowercase alphanumeric string (with underscores permitted) that conforms to the extraction rule defined in Constraints, and future datasets may introduce context values not enumerated here. **Migration note (v1.1):** The literal token `options` appears in this historical list. Under the v1.1 split rule, any element whose `name` ends in `_options` is routed to `ans_image_filename` (Field 2.15); future re-derivations of the dataset will therefore not record the token in `qn_image_context`. Pre-v1.1 datasets that pre-classify an `options` token here remain valid.

---

### 2.12 Answer Options

| Property | Value |
|---|---|
| Internal Key | `answer_options` |
| Format | UTF-8 string encoded in Markdown syntax; nullable **only** under the explicit sentinel convention described below |
| Constraints | Enumerates all candidate answer choices for the question, preserving the original ordering and textual content from the source paper. Three mutually exclusive modalities are recognised from the joint contents of this field and `ans_image_filename` (§2.15): |

**Answer-Option Modalities (v1.2):**

| Modality | `answer_options` content | `ans_image_filename` content | Example |
|---|---|---|---|
| `text_only` | Markdown block containing `A. …` through `D. …` | empty | 2015 Q1, Q2, Q3 … |
| `image_only` | The literal sentinel `[IMAGE]` (square-bracketed uppercase token, no surrounding whitespace) | non-empty (semicolon-delimited list, typically one entry) | 2015 Q18 |
| `image_plus_text` | Markdown block containing `A. …` through `D. …` (verbatim, no alteration for the accompanying image) | non-empty (semicolon-delimited list of auxiliary images) | 2015 Q25 |

**Sentinel handling:** The literal token `[IMAGE]` is reserved exclusively for the `image_only` modality. **Must** substitute the natural `A.`/`B.`/`C.`/`D.` markdown block with the literal sentinel `[IMAGE]` when no textual options are present in the OCR output but at least one `_options`-suffixed element exists. Conversely, when both textual options and an option-block image are present, the textual options are preserved verbatim and the image is recorded in the `ans_image_filename` column.

---

### 2.13 Correct Answer

| Property | Value |
|---|---|
| Internal Key | `correct_answer` |
| Format | Single uppercase alphabetic character, or the literal `*` token |
| Constraints | Admissible values are drawn exclusively from the set `{A, B, C, …, Z}`, and must correspond to one of the options enumerated in `answer_options`. The literal token `*` is a reserved value indicating a withdrawn/cancelled question whose answer key is not published (e.g., 2018 Paper 1A Q40). |

---

### 2.14 Cohort Percentage

| Property | Value |
|---|---|
| Internal Key | `cohort_pct` |
| Format | Integer in the inclusive range `0–100`; nullable / empty value permitted |
| Constraints | **Optional supplementary field.** Records the cohort performance percentage (i.e., the percentage of sitting candidates who answered the question correctly) as printed on the official answer-key sheet next to the letter choice. Extracted in Stage 2 from the parenthetical suffix of the `Key` cell (e.g., `"C (51%)"` → `51`). If the printed answer key omits the cohort percentage for a given year or question, this field **must** be left empty (blank/NA) rather than defaulted to `0`. |

**Examples:**

| Source Answer-Key Cell | Parsed `correct_answer` | Parsed `cohort_pct` |
|---|---|---|
| `A (40%)` | `A` | `40` |
| `C` (no percentage printed) | `C` | *(empty / NA)* |

---

### 2.15 Answer-Option Image Filename

| Property | Value |
|---|---|
| Internal Key | `ans_image_filename` |
| Format | Semicolon-delimited list of URI-formatted strings (hyperlinks or relative/absolute file paths); nullable / empty value permitted |
| Constraints | References one or more graphical assets embedded within the **answer-option block** of `answer_options` (i.e., visually positioned within or around the `A./B./C./D.` choices, or — for the `image_only` modality — the entire graphical options themselves). Each list item corresponds 1:1 to an entry in the `elements` array of the per-year crop-manifest JSON whose `name` ends with the literal suffix `_options`, in manifest order. **Independence rule:** the cardinality of this field is not required to equal the cardinality of the `qn_image_*` family (Fields 2.9–2.11); the two are independent. |

**Modality mapping (jointly with §2.12 `answer_options`):**

| Modality | `answer_options` | `ans_image_filename` | Meaning |
|---|---|---|---|
| `text_only` | `A.`/`B.`/`C.`/`D.` text block | **empty** | Conventional textual options, no option-image. |
| `image_only` | Literal sentinel `[IMAGE]` | **non-empty** | The image(s) constitute the entire option set; consumers must render them to read the choices. |
| `image_plus_text` | `A.`/`B.`/`C.`/`D.` text block (preserved verbatim) | **non-empty** | An auxiliary image supplements but does not replace the textual options. |

**Dispatch Rule Summary (for v1.1 and later):**

| Source element `name` suffix | Routed to |
|---|---|
| ends in literal `_options` | `ans_image_filename` (Field 2.15) |
| any other suffix | `qn_image_*` family (Fields 2.9–2.11) |

**Observed element name examples:** `2015DSE1A_q18_options`, `2015DSE1A_q25_options` — both routed to `ans_image_filename`; `2015DSE1A_q25_design_view` — routed to the `qn_image_*` family.

---

### 2.16 Cluster Flag

| Property | Value |
|---|---|
| Internal Key | `cluster` |
| Format | Integer flag: `0` or `1` |
| Constraints | Indicates whether the question belongs to a **cluster** — a group of two or more questions that share a common context (a passage, diagram, table, or algorithm) introduced by an "Answer Questions X and Y with reference to …" preamble. `1` marks a cluster member; `0` marks a standalone question. All members of the same cluster share the identical `cluster_desc` (Field 2.17). |

---

### 2.17 Cluster Description

| Property | Value |
|---|---|
| Internal Key | `cluster_desc` |
| Format | UTF-8 string encoded in Markdown syntax; empty when `cluster = 0` |
| Constraints | The **verbatim** shared context (the "Answer Questions … with reference to …" intro sentence, any image links, and any fenced code) that introduces a cluster. For a cluster member (`cluster = 1`), this field holds the full preamble text and is identical across all members of the cluster. For a standalone question (`cluster = 0`), this field **must** be left empty (blank / NA). |

**Example (2023 Q28/Q29):** the value begins with the intro line `Answer Questions 28 and 29 with reference to the following algorithms.`, followed by the image link `![Q28 algorithms](2023DSE1A_images/2023DSE1A_q28_algorithms.png)` and the fenced `Algorithm 1` / `Algorithm 2` code block — reproduced verbatim from the source Markdown.
